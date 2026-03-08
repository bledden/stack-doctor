"""Tiny inference server: loads Qwen2.5-1.5B via MLX, exposes /generate endpoint.

Run alongside the environment server:
    .venv-infer/bin/python inference_server.py

The dashboard calls POST /generate with {prompt, max_tokens} and gets back {text}.
"""

import json
import re
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from mlx_lm import load, generate

print("Loading Qwen2.5-1.5B-Instruct (4-bit MLX)...")
t0 = time.time()
MODEL, TOKENIZER = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
print(f"Model loaded in {time.time()-t0:.1f}s — inference server ready on :8001")

SYSTEM_PROMPT = """You are Stack Doctor, an expert AI agent that diagnoses inference-stack incidents.
You receive an incident ticket with hardware/model/backend context, log excerpts, and specialist opinions.
Some specialists may be wrong. Output a JSON array of actions:
  {"type":"inspect","target":"logs|config|snippet|metrics"}
  {"type":"ask_specialist","specialist":"runtime|dispatch|kernel|loader"}
  {"type":"apply_fix","fix":"<fix_name>"}
  {"type":"submit","root_cause":"<cause>","fix":"<fix>","justification":"<why>"}"""


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/generate":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            user_msg = body.get("prompt", "")
            max_tokens = body.get("max_tokens", 512)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            # Add conversation history if provided
            history = body.get("history", [])
            if history:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

            prompt = TOKENIZER.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            t0 = time.time()
            text = generate(MODEL, TOKENIZER, prompt=prompt, max_tokens=max_tokens, verbose=False)
            gen_time = time.time() - t0
            print(f"Generated {len(text)} chars in {gen_time:.1f}s")

            resp = json.dumps({"text": text, "gen_time": gen_time})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", len(resp))
            self.end_headers()
            self.wfile.write(resp.encode())
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Quiet logs


HTTPServer(("0.0.0.0", 8001), Handler).serve_forever()
