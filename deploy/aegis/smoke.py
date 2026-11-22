#!/usr/bin/env python3
"""Fixed requests against a running Strata server (temperature 0), one JSON line per request to <out>/results.jsonl.

    smoke.py <base-url> <outdir> [--long-file PATH]          SMOKE_ONLY=chat,code limits the requests, SMOKE_REPEAT=N repeats them,
                                                             SMOKE_NONCE=1 makes every prompt fresh

  chat   a short explanation              (decode speed, greedy text for card-to-card comparison)
  code   a small Python function          (decode speed on code, where the MTP draft hits more)
  long   ~4K tokens of source + question  (prompt speed)
  long16 ~16K tokens of source + question (opt-in via SMOKE_ONLY)
  deep   SMOKE_DEEP_TOKENS=N: ~N tokens of serve/server.py with a needle at 40% (opt-in: SMOKE_ONLY=...,deep);
         content must name BLUE-HERON-7342
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REQS = [
    ("chat", "Explain in exactly three sentences what a mutex is and when to use one.", 400),
    ("code", "Write a Python function `fib(n)` that returns the n-th Fibonacci number iteratively. Include a docstring "
             "and three doctest examples. Reply with the code only.", 500),
]


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data, time.time() - t0


def main():
    base, out = sys.argv[1].rstrip("/"), Path(sys.argv[2])
    long_file = Path(sys.argv[sys.argv.index("--long-file") + 1]) if "--long-file" in sys.argv else None
    reqs = list(REQS)
    if long_file:
        src = long_file.read_text(errors="replace")[:15000]          # ~4K tokens of code
        reqs.append(("long", f"Here is a source file:\n\n```\n{src}\n```\n\nIn two sentences: what does this file do?", 200))
        big = long_file.read_text(errors="replace")[:60000]          # ~16K tokens (opt-in: SMOKE_ONLY=...,long16)
        reqs.append(("long16", f"Here is a source file:\n\n```\n{big}\n```\n\nList the five most important functions and what each does.", 300))
    deep_tokens = int(os.environ.get("SMOKE_DEEP_TOKENS", "0"))          # opt-in: a long-context needle test
    if deep_tokens:
        hay = (Path(__file__).resolve().parents[2] / "terminal" / "serve" / "server.py").read_text(errors="replace")
        hay = hay[:int(deep_tokens * 3.5)]                              # ~3.5 characters per token of Python
        cut = hay.rfind("\n", 0, int(len(hay) * 0.4)) + 1               # the needle at ~40% of the prompt
        hay = hay[:cut] + "# NOTE: the deploy codename for this server is BLUE-HERON-7342.\n" + hay[cut:]
        reqs.append(("deep", f"Here is a source file:\n\n```\n{hay}\n```\n\nOne comment in the file starts with "
                             "'# NOTE:' and names a deploy codename. Reply with that codename only.", 400))
    only = [x for x in os.environ.get("SMOKE_ONLY", "").split(",") if x]     # e.g. SMOKE_ONLY=chat
    if only:
        reqs = [r for r in reqs if r[0] in only]
    else:
        reqs = [r for r in reqs if r[0] not in ("long16", "deep")]  # the long prompts only when asked for
    reqs = reqs * max(1, int(os.environ.get("SMOKE_REPEAT", "1")))      # SMOKE_REPEAT=3: the list three times in a row
    with open(out / "results.jsonl", "a") as f:
        for name, prompt, max_tokens in reqs:
            if os.environ.get("SMOKE_NONCE") == "1":                  # a fresh prefix every time: no prompt-cache hit
                prompt = f"[run {os.urandom(6).hex()}] {prompt}"
            body = {"model": "strata", "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                    "max_tokens": max_tokens, "stream": False}
            try:
                data, wall = post(base + "/v1/chat/completions", body)
            except Exception as e:                                      # keep going: one failed request is a result too
                rec = {"name": name, "error": repr(e)}
                print(json.dumps(rec), flush=True)
                f.write(json.dumps(rec) + "\n")
                continue
            msg = data["choices"][0]["message"]
            rec = {"name": name, "wall_s": round(wall, 2), "usage": data.get("usage"), "timings": data.get("timings"),
                   "finish": data["choices"][0].get("finish_reason"),
                   "reasoning": (msg.get("reasoning_content") or "")[:2000], "content": (msg.get("content") or "")[:4000]}
            f.write(json.dumps(rec) + "\n")
            t = rec["timings"] or {}
            print(f"{name:5s} wall {wall:6.1f}s  usage {rec['usage']}  "
                  f"pp {t.get('prompt_per_second', '-')}  tg {t.get('predicted_per_second', '-')}  finish {rec['finish']}",
                  flush=True)


if __name__ == "__main__":
    main()
