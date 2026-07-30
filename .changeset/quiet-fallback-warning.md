---
"@lfm-encoder/core": patch
---

Stop transformers.js warning about `TwoTowerGraph` on every load. The exported graph is a plain
named-inputs/named-outputs session, so `resolve_model_type` cannot place it and falls back to
`EncoderOnly` — which is exactly the shape this loader wants. The fallback was correct and the warning
was noise, so `loadEncoderModel` now sets `env.logLevel = LogLevel.ERROR`; real failures still surface.
