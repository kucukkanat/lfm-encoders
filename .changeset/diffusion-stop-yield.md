---
"@lfm-encoder/tasks": patch
---

Make `signal` actually able to interrupt a diffusion generation. The decode loop only ever awaited the
forward pass, which resumes in a microtask — so in a single-threaded host a `postMessage` asking it to
stop stayed undelivered until the loop had already finished. The loop now yields a macrotask each pass
before re-checking `signal.aborted`, costing microseconds against a pass measured in hundreds of
milliseconds.
