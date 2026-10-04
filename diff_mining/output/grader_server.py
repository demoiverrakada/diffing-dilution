
import asyncio, os, time, uuid, torch, uvicorn
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL, DEVICE = "Qwen/Qwen3-4B-Instruct-2507", os.environ["GRADER_DEVICE"]
MAX_BATCH, MAX_NEW = 4, 3072  # batch 16 x 4096 ran out of GPU memory in the ADL toolkit run
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEVICE).eval()
app, queue = FastAPI(), asyncio.Queue()

def flat(messages):  # the coherence grader sends OpenAI content-part lists; Qwen's template needs strings
    return [{**m, "content": m["content"] if isinstance(m["content"], str) else
             "".join(p.get("text", "") for p in m["content"] if isinstance(p, dict))} for m in messages]

def generate_batch(items):
    texts = [tok.apply_chat_template(flat(it["messages"]), add_generation_prompt=True, tokenize=False) for it in items]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
    temps = [it["temperature"] for it in items]
    kw = dict(max_new_tokens=min(MAX_NEW, max(it["max_tokens"] for it in items)), pad_token_id=tok.pad_token_id)
    if all(t == 0 for t in temps):
        kw["do_sample"] = False
    else:
        kw.update(do_sample=True, temperature=max(0.1, sum(temps) / len(temps)), top_p=0.95)
    with torch.no_grad():
        out = model.generate(**enc, **kw)
    return [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in out]

async def worker():
    loop = asyncio.get_running_loop()
    while True:
        items = [await queue.get()]
        await asyncio.sleep(0.05)
        while not queue.empty() and len(items) < MAX_BATCH:
            items.append(queue.get_nowait())
        try:
            for it, text in zip(items, await loop.run_in_executor(None, generate_batch, items)):
                it["fut"].set_result(text)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for it in items:  # retry one at a time
                try:
                    it["fut"].set_result((await loop.run_in_executor(None, generate_batch, [it]))[0])
                except Exception as e:
                    torch.cuda.empty_cache(); it["fut"].set_exception(e)
        except Exception as e:
            for it in items:
                it["fut"].set_exception(e)

@app.on_event("startup")
async def startup():
    asyncio.create_task(worker())

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    t = body.get("temperature")
    fut = asyncio.get_running_loop().create_future()
    await queue.put({"messages": body["messages"], "fut": fut, "temperature": 1.0 if t is None else float(t),
                     "max_tokens": int(body.get("max_tokens") or body.get("max_completion_tokens") or 512)})
    text = await fut
    return {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
            "model": body.get("model", "local"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
