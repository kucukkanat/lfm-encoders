# @lfm-encoder/core

Loading, tokenization with character offsets, and span pooling for the
[LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M) ONNX exports.

This is the layer under [`@lfm-encoder/tasks`](../tasks/README.md). Reach for it when you want the raw
graph — sentence embeddings, your own head, a task the wrappers do not cover. If you just want zero-shot
routing or policy linting, use `@lfm-encoder/tasks` instead.

```bash
bun add @lfm-encoder/core @huggingface/transformers
```

`@huggingface/transformers` (v4.2+) is a peer dependency: one copy of it has to own the onnxruntime session
and the model cache.

## `loadEncoderModel`

```ts
loadEncoderModel(id: string, options?: LoadOptions): Promise<EncoderModel>
```

```ts
import { loadEncoderModel } from "@lfm-encoder/core";

const model = await loadEncoderModel("LFM2.5-Encoder-350M-ONNX", {
	modelRoot: "./models", // a URL prefix in the browser, a path in Node; omit to use the Hub
	dtype: "q8",
	onProgress: ({ file, fraction }) => {
		if (fraction !== undefined) console.log(`${file} ${Math.round(fraction * 100)}%`);
	},
});

model.config;
// { task: "fill-mask", sourceModel: "LiquidAI/LFM2.5-Encoder-350M", hiddenSize: 1024,
//   vocabSize: 65536, realVocabSize: 64402, outputs: ["logits", "last_hidden_state"] }

const tokenized = model.tokenize("The capital of France is <|mask|>.");
const outputs = await model.forward(tokenized);

const hidden = outputs.last_hidden_state;
if (hidden === undefined) throw new Error("no last_hidden_state");
console.log(hidden.rows, hidden.cols); // 9 1024  — one row per token

await model.dispose();
```

`forward` returns `Readonly<Record<string, Matrix>>` keyed by the graph's output names, which are listed in
`model.config.outputs`. Indexing it yields `Matrix | undefined`, so narrow before use — that is the type
telling you a `token_proj` lookup on the base encoder would be a mistake.

### `LoadOptions`

| Field | Default | Notes |
| --- | --- | --- |
| `modelRoot` | Hugging Face Hub | URL prefix in the browser, filesystem path in Node. Sets a module-level singleton inside transformers.js, so a process has exactly one root. |
| `dtype` | `"q8"` | `"fp32" \| "q8" \| "q4"` are built by default. See the [accuracy and speed table](../../README.md#accuracy-and-speed) before changing it — `fp32` is exact and faster than `q8`; `q8` is the default only because it is 4× smaller to ship. |
| `device` | `"auto"` | `"auto"` picks WebGPU when the browser exposes it, WASM otherwise. Node always gets WASM. |
| `onProgress` | — | Called per download chunk. `fraction` is `undefined` until the server reports a content length. |

### `EncoderModel`

| Member | Type |
| --- | --- |
| `id` | `string` |
| `dtype` | `Dtype` |
| `config` | `ModelConfig` — the subset of `config.json` this library relies on, plus `head` on the task repos |
| `tokenizer` | `PreTrainedTokenizer` from transformers.js |
| `tokenize(text)` | `Tokenized`. Cheap; no session involved. |
| `forward(tokenized)` | `Promise<Readonly<Record<string, Matrix>>>` |
| `dispose()` | `Promise<void>` |

### `Matrix`

`forward` hands back `(tokens, features)` views over a flat `Float32Array`. The leading batch axis is
dropped — every task here runs a single sequence.

```ts
interface Matrix {
	readonly rows: number;
	readonly cols: number;
	readonly data: Float32Array;
	row(index: number): Float32Array; // zero-copy subarray
}
```

## `tokenizeWithSpans`

```ts
tokenizeWithSpans(tokenizer: PreTrainedTokenizer, text: string): Tokenized
```

`model.tokenize` is this function bound to the model's tokenizer; call it directly when you have a
tokenizer but no session.

**Why it exists.** transformers.js has no `return_offsets_mapping`. Both task heads pool over *character
ranges* — "the tokens covering rule 2", "the tokens after the prefix" — so without offsets there is no way
to run them at all. This reconstructs the mapping.

The reconstruction is exact, not heuristic. The tokenizer is a plain byte-level BPE with no normalizer and
no byte fallback, so token strings concatenate back to the input losslessly, and in that alphabet every
character stands for exactly one UTF-8 byte. Token *byte* lengths are therefore just string lengths, and a
running sum gives byte offsets that get translated into JavaScript string indices. The obvious
alternative — `decode()` each id and accumulate lengths — is wrong for exactly the languages this model is
for: a token holding a partial UTF-8 sequence decodes to `U+FFFD` and offsets drift the moment text stops
being ASCII. The integration tests assert this matches HuggingFace's `return_offsets_mapping` byte for
byte on every fixture, including the non-ASCII ones (German, French, Spanish).

```ts
import { tokenizeWithSpans, tokensIn } from "@lfm-encoder/core";

const text = "Ich möchte eine Rückerstattung.";
const { ids, spans } = tokenizeWithSpans(model.tokenizer, text);

ids.length; // 9
spans[0]; // { start: 0, end: 0 }  — BOS, inserted by the tokenizer, so zero-width
spans[1]; // { start: 0, end: 3 }  — "Ich"

// Which tokens cover the word "Rückerstattung"?
for (const i of tokensIn(spans, { start: 16, end: 30 })) {
	console.log(i, JSON.stringify(text.slice(spans[i]?.start, spans[i]?.end)));
}
// 4 " Rück"
// 5 "erst"
// 6 "att"
// 7 "ung"
```

Spans are half-open, in JavaScript string indices (UTF-16 code units), so `text.slice(start, end)` is
always the right way back to the source.

### `tokensIn`

```ts
tokensIn(spans: readonly Span[], span: Span): number[]
```

Indices of the tokens overlapping `span`. Zero-width tokens never match, which is what keeps BOS and other
inserted specials out of every pool — the same rule the reference implementation applies.

## Pooling

```ts
meanRows(matrix: Matrix, indices: readonly number[]): Float32Array
poolSpan(projections: Matrix, spans: readonly Span[], span: Span): Float32Array
normalize(vector: Float32Array): Float32Array
dot(a: Float32Array, b: Float32Array): number
softmax(logits: readonly number[]): number[]
sigmoid(value: number): number
```

`poolSpan` is `meanRows(projections, tokensIn(spans, span))`. `meanRows` returns zeros for an empty
selection, matching the reference behaviour for a label that tokenizes to nothing. `normalize` clamps the
norm at `1e-12`; `softmax` is max-shifted.

Pooling *after* projection is exact rather than an approximation — both task towers are affine, and an
affine map commutes with a mean, so `W @ mean(h) + b == mean(W @ h + b)`. That reassociation is why the
ONNX graph can emit per-token projections and stay independent of how many labels the caller passes. See
the [root README](../../README.md#how-the-zero-shot-heads-work).

## Prompt building

```ts
buildPrefix(heading: string, labels: readonly string[]): string
labelRanges(heading: string, labels: readonly string[]): Span[]
textSpan(prefix: string, text: string): Span
```

The exact prompt layout is what the task heads were trained on, and the character arithmetic in
`labelRanges` depends on it byte for byte. Use these rather than formatting the string yourself.

```ts
import { buildPrefix, labelRanges, textSpan } from "@lfm-encoder/core";

const labels = ["billing", "technical support"];
const prefix = buildPrefix("Categories", labels);
// "Categories:\n- billing\n- technical support\n\nText:\n"

const ranges = labelRanges("Categories", labels);
// [{ start: 14, end: 21 }, { start: 24, end: 41 }]
ranges.map((r) => prefix.slice(r.start, r.end)); // ["billing", "technical support"]

textSpan(prefix, "refund please"); // { start: 49, end: 62 }
```

The heading is `"Categories"` for the router and `"Policy"` for the linter; both are recorded as
`model.config.head.prefixHeading`, so read it off the model rather than hard-coding it. An empty label list
emits `- (none)`.

## Bring your own head

The base encoder's `last_hidden_state` is a general-purpose 1024-d representation. Mean-pool it and you
have sentence embeddings — no task wrapper involved.

```ts
import { dot, loadEncoderModel, meanRows, normalize, tokensIn } from "@lfm-encoder/core";

const model = await loadEncoderModel("LFM2.5-Encoder-350M-ONNX", { modelRoot: "./models" });

/** Mean-pool the real tokens into one L2-normalised sentence vector. */
async function embed(text: string): Promise<Float32Array> {
	const tokenized = model.tokenize(text);
	const outputs = await model.forward(tokenized);
	const hidden = outputs.last_hidden_state;
	if (hidden === undefined) throw new Error("encoder produced no last_hidden_state");

	// Spanning the whole text selects every token except the zero-width specials.
	const real = tokensIn(tokenized.spans, { start: 0, end: text.length });
	return normalize(meanRows(hidden, real));
}

const [a, b, c] = await Promise.all([
	embed("How do I reset my password?"),
	embed("I forgot my login credentials."),
	embed("The mitochondrion is the powerhouse of the cell."),
]);

// Both vectors are unit length, so `dot` is cosine similarity.
dot(a, b); // 0.742
dot(a, c); // 0.589

await model.dispose();
```

Two things worth knowing before you build on this:

- The base checkpoint is a masked-LM, not a contrastively trained embedding model. Cosine similarities are
  compressed into a narrow band — the 0.74 vs 0.59 above is a real signal but a weak one. Calibrate
  thresholds on your own data; do not assume the scale of a sentence-transformers model.
- The `q8` numbers above will move under a different `dtype`. Embeddings inherit the quantization error in
  the [accuracy table](../../README.md#accuracy-and-speed) without a softmax to squash it.

To score against a *fixed* label set with real separation, use the router head in `@lfm-encoder/tasks` —
it was trained for exactly that and needs one pass, not one per label.

## Types

`Device`, `Dtype`, `HeadConfig`, `LoadOptions`, `LoadProgress`, `Matrix`, `ModelConfig`, `Span`,
`Tokenized` are all exported.

`HeadConfig` is present only on the task repos and carries everything needed to reproduce a head in JS:

```ts
model.config.head;
// { kind: "cosine", normalize: true, scale: 1.3714938163757324,
//   bias: -0.2723352313041687, activation: "softmax",
//   prefixHeading: "Categories", projDim: 256 }
```
