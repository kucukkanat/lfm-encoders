---
"@lfm-encoder/core": minor
"@lfm-encoder/tasks": minor
---

Initial release: LFM2.5-Encoder-350M traced to ONNX and run client-side through transformers.js, with
zero-shot prompt routing, policy linting, fill-mask and masked diffusion. Includes exact
character-offset reconstruction for
the byte-level BPE (transformers.js exposes no `return_offsets_mapping`) and span pooling that reassociates
the reference heads' pool-then-project into project-then-pool, which keeps one static graph usable for any
label set. Masked diffusion adds a full MDLM decode loop — block-scheduled, confidence-ordered,
adjacency-aware — over the same static graph, with per-pass frame callbacks for animating the denoising.
