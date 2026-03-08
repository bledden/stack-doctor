"""Unified server for HF Spaces: environment + inference + dashboard on port 7860."""

import json
import os
import sys
import time
import threading

sys.path.insert(0, "/app")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from server.app import app as env_app

env_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model state (loaded in background)
MODEL_STATE = {"model": None, "tokenizer": None, "ready": False, "error": None}

UNTRAINED_SYSTEM = (
    "You are Stack Doctor, an expert AI agent that diagnoses inference-stack incidents.\n"
    "You receive an incident ticket with hardware/model/backend context, log excerpts, and specialist opinions.\n"
    "Some specialists may be wrong. Output a JSON array of actions:\n"
    '  {"type":"inspect","target":"logs|config|snippet|metrics"}\n'
    '  {"type":"ask_specialist","specialist":"runtime|dispatch|kernel|loader"}\n'
    '  {"type":"apply_fix","fix":"<fix_name>"}\n'
    '  {"type":"submit","root_cause":"<cause>","fix":"<fix>","justification":"<why>"}'
)

TRAINED_SYSTEM = (
    "You are Stack Doctor, an expert AI agent that diagnoses inference-stack incidents.\n"
    "You are methodical: first inspect logs and config, then query specialists to cross-verify (some lie), then apply a fix and submit.\n\n"
    "Available actions (output as a JSON array):\n"
    '  {"type":"inspect","target":"logs"} or "config" or "snippet" or "metrics"\n'
    '  {"type":"ask_specialist","specialist":"runtime"} or "dispatch" or "kernel" or "loader"\n'
    '  {"type":"apply_fix","fix":"<name>"} -- available fixes: add_whitelist_entry, fix_comm_config, fix_quantization, fix_runtime_path, fix_weight_mapping, relax_arch_check, switch_backend, tune_memory_config, update_driver_config, update_model_config\n'
    '  {"type":"submit","root_cause":"<cause>","fix":"<fix>","justification":"<detailed reasoning>"}\n\n'
    "Available root causes: arch_guard, backend_selector, backend_whitelist, distributed_comm, driver_compat, memory_oom, model_config, quantization_error, runtime_loader, weight_layout\n\n"
    "IMPORTANT: Pick ONE target per inspect, ONE specialist per query. Investigate before submitting. Give a detailed justification.\n\n"
    "Example output:\n"
    '[{"type":"inspect","target":"logs"},{"type":"inspect","target":"config"},{"type":"ask_specialist","specialist":"kernel"},'
    '{"type":"apply_fix","fix":"relax_arch_check"},'
    '{"type":"submit","root_cause":"arch_guard","fix":"relax_arch_check","justification":"Logs show architecture check failure for SM90. Config confirms guard enabled. Kernel specialist confirmed not a kernel issue."}]'
)


def load_model_background():
    """Load Qwen 1.5B in a background thread so the server starts fast."""
    try:
        print("[Model] Loading Qwen2.5-1.5B-Instruct (CPU)...")
        t0 = time.time()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

        MODEL_STATE["model"] = model
        MODEL_STATE["tokenizer"] = tokenizer
        MODEL_STATE["ready"] = True
        print(f"[Model] Loaded in {time.time()-t0:.1f}s")
    except Exception as ex:
        MODEL_STATE["error"] = str(ex)
        print(f"[Model] Failed to load: {ex}")


threading.Thread(target=load_model_background, daemon=True).start()


@env_app.post("/generate")
async def generate_endpoint(request: Request):
    body = await request.json()
    prompt_text = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 512)
    mode = body.get("mode", "untrained")

    if not MODEL_STATE["ready"]:
        if MODEL_STATE["error"]:
            return JSONResponse({"error": MODEL_STATE["error"]}, status_code=500)
        return JSONResponse({"error": "Model still loading, please wait..."}, status_code=503)

    model = MODEL_STATE["model"]
    tokenizer = MODEL_STATE["tokenizer"]
    system = TRAINED_SYSTEM if mode == "trained" else UNTRAINED_SYSTEM

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt_text},
    ]

    import torch

    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text_input, return_tensors="pt")

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    gen_time = time.time() - t0
    print(f"[Model] Generated {len(text)} chars in {gen_time:.1f}s (mode={mode})")
    return JSONResponse({"text": text, "gen_time": gen_time})


@env_app.get("/model_status")
async def model_status():
    return JSONResponse({
        "ready": MODEL_STATE["ready"],
        "error": MODEL_STATE["error"],
    })


@env_app.get("/", include_in_schema=False)
async def root():
    return FileResponse("/app/static/index.html")


if __name__ == "__main__":
    uvicorn.run(env_app, host="0.0.0.0", port=7860)
