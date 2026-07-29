# @lfm-encoder/tasks

Zero-shot prompt routing, policy linting and fill-mask on
[LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M), running fully client-side.

Reproduces Liquid's [prompt-routing](https://huggingface.co/spaces/LiquidAI/prompt-routing) and
[policy-linting](https://huggingface.co/spaces/LiquidAI/policy-linting) Spaces. Both zero-shot tasks take
free-text labels supplied at call time — labels are ordinary prose, can change on every call, and nothing
is trained or cached per label set.

```bash
bun add @lfm-encoder/tasks @huggingface/transformers
```

Every loader takes the same [`LoadOptions`](../core/README.md#loadoptions) as
[`@lfm-encoder/core`](../core/README.md): `modelRoot` (URL prefix in the browser, path in Node, omit for
the Hub), `dtype` (default `q8` — see the [accuracy table](../../README.md#accuracy-and-speed)), `device`,
`onProgress`. All examples below run at `q8` and their outputs are real.

## `loadPromptRouter`

```ts
loadPromptRouter(options?: LoadOptions): Promise<PromptRouter>
route(text: string, labels: readonly string[]): Promise<RouteResult>
```

Scores one text against a set of categories. Cosine head, softmax across labels — the scores are
**comparative** and sum to 1, so they answer "which of these" and not "does this match at all".

```ts
import { loadPromptRouter } from "@lfm-encoder/tasks";

const router = await loadPromptRouter({ modelRoot: "./models" });

const categories = [
	"billing and payments",
	"technical support",
	"account management",
	"sales enquiry",
];

const result = await router.route(
	"The deployment keeps crashing with an out-of-memory error on startup.",
	categories,
);

result.top; // { label: "technical support", score: 0.8376356907485364 }
result.tokenCount; // 40 — the categories *and* the text, in one pass

for (const { label, score } of result.scores) {
	console.log(`${(score * 100).toFixed(1).padStart(5)}%  ${label}`);
}
// 83.8%  technical support
//  5.4%  account management
//  5.4%  billing and payments
//  5.4%  sales enquiry

await router.dispose();
```

`scores` is sorted descending and `top` is `scores[0]`. Empty label sets throw rather than guess.

Because the softmax is over the labels you passed, the same text re-scores when the set changes — that is
the feature, not a wart:

```ts
const text = "how do I reset my password";
(await router.route(text, ["account access", "billing"])).top.label; // "account access"
(await router.route(text, ["cooking recipes", "billing"])).top.label; // "billing"
```

If you need an *absolute* "none of these" signal, add an explicit escape-hatch category (`"unrelated"`,
`"small talk"`) instead of thresholding the probabilities. A three-way tie at 5.4% is the model telling you
none of the losers fit; it is not a calibrated rejection score.

| Type | Shape |
| --- | --- |
| `RouteScore` | `{ label: string; score: number }` |
| `RouteResult` | `{ top: RouteScore; scores: readonly RouteScore[]; tokenCount: number }` |
| `PromptRouter` | `{ model: EncoderModel; route(...); dispose() }` |

## `loadPolicyLinter`

```ts
loadPolicyLinter(options?: LoadOptions): Promise<PolicyLinter>
lint(text: string, rules: readonly string[]): Promise<LintResult>
```

Scores **every word against every rule** in a single pass. Dot head, sigmoid per pair — scores are
independent and absolute, so they answer "does this word violate this rule" and do not sum to anything.

```ts
import { loadPolicyLinter } from "@lfm-encoder/tasks";

const linter = await loadPolicyLinter({ modelRoot: "./models" });

const draft = "We guarantee a 300% return within six months, risk free.";
const policy = [
	"no guarantees about financial returns",
	"no medical or health claims",
	"no disparaging competitors by name",
];

const report = await linter.lint(draft, policy);

for (const flag of report.flagged()) {
	console.log(`${flag.score.toFixed(2)}  ${flag.text.padEnd(8)} ${flag.rule}`);
}
// 0.98  300%     no guarantees about financial returns
// 0.93  return   no guarantees about financial returns
// 0.78  risk     no guarantees about financial returns
// 0.68  free.    no guarantees about financial returns

report.tokenCount; // 41
await linter.dispose();
```

### Rendering flags back onto the text

`LintWord` and `LintFlag` both extend `Span`, and `start`/`end` are character offsets **into the original
`text`** — the prompt prefix is already subtracted. `text.slice(flag.start, flag.end)` is always the source
substring, so highlighting is plain string arithmetic.

```ts
import type { LintFlag } from "@lfm-encoder/tasks";

/** Wrap every flagged word. Swap the brackets for a <mark> in a real UI. */
function highlight(text: string, flags: readonly LintFlag[]): string {
	let out = "";
	let cursor = 0;
	// flagged() is worst-first; render left-to-right instead.
	for (const flag of [...flags].sort((a, b) => a.start - b.start)) {
		if (flag.start < cursor) continue; // one word can trip several rules — mark it once
		out += `${text.slice(cursor, flag.start)}[${text.slice(flag.start, flag.end)}]`;
		cursor = flag.end;
	}
	return out + text.slice(cursor);
}

highlight(draft, report.flagged());
// "We guarantee a [300%] [return] within six months, [risk] [free.]"
```

The `flag.start < cursor` skip is load-bearing: a word that violates two rules appears twice in `flagged()`,
once per rule. To colour by rule instead, group first:

```ts
const byRule = policy.map((rule, index) => ({
	rule,
	hits: report.flagged().filter((flag) => flag.ruleIndex === index),
}));
```

`report.words` holds *every* word in order — flagged or not — each with one score per rule in the order the
rules were supplied. Use it to draw a heatmap or pick your own aggregation:

```ts
report.words[0]; // { text: "We", start: 0, end: 2, scores: [0.0013, 4.3e-9, 2.6e-8] }

// Worst violation of rule 1 anywhere in the draft.
const worst = Math.max(...report.words.map((word) => word.scores[1] ?? 0));
```

### Threshold

```ts
export const DEFAULT_THRESHOLD = 0.5;
report.flagged(threshold?: number): LintFlag[]
```

`flagged()` filters `report.words` — it does **not** re-infer. Sweeping the threshold in a UI slider is
free once `lint()` has returned.

`0.5` is the neutral sigmoid midpoint and matches the reference UI. It is not a calibrated operating point,
and the right value is task-dependent:

| Threshold | Effect |
| --- | --- |
| Lower (0.2–0.4) | Catches hedged and indirect phrasing. Also catches ordinary words adjacent to a violation — the model tends to smear a violation across its whole clause, so context words come along. Right for review queues where a human adjudicates. |
| `0.5` (default) | What the Space uses. |
| Higher (0.7–0.9) | Only the words that carry the violation. On the example above, `0.8` keeps `300%` and `return` and drops `risk free.`. Right for automated blocking, where a false positive is expensive. |

Two things to hold in mind when tuning. Scores on this head are strongly bimodal — most words sit at 1e-7
and violations sit above 0.6 — so moving the threshold between 0.3 and 0.5 often changes nothing at all.
And `q8` moves 3 of the 192 word/rule decisions in the fixtures across the 0.5 line versus fp32 (see the
[accuracy table](../../README.md#accuracy-and-speed)), so a threshold tuned to within a few hundredths is tuned to
quantization noise, not to the model.

| Type | Shape |
| --- | --- |
| `LintWord` | `{ text: string; start: number; end: number; scores: readonly number[] }` |
| `LintFlag` | `{ text; start; end; ruleIndex: number; rule: string; score: number }` |
| `LintResult` | `{ rules; words: readonly LintWord[]; tokenCount: number; flagged(threshold?) }` |
| `PolicyLinter` | `{ model: EncoderModel; lint(...); dispose() }` |

**Words, not tokens.** A byte-level BPE splits `guarantee` into pieces that each carry their own score, and
highlighting half a word reads as a bug. Words are `/\S+/gu` matches and take the **max** over their
pieces, which is what the reference UI does. Note that trailing punctuation rides along: `free.` and
`Corp,` are single words here.

## `loadFillMask`

```ts
loadFillMask(options?: LoadOptions): Promise<FillMask>
predict(text: string, options?: { topK?: number }): Promise<MaskSlot[]>
```

The base encoder with its tied masked-LM head. Bidirectional, so each mask is conditioned on the whole
sentence rather than only its left context, and every mask in the input is filled in one pass.

```ts
import { loadFillMask } from "@lfm-encoder/tasks";

const filler = await loadFillMask({ modelRoot: "./models" });
filler.maskToken; // "<|mask|>"

const slots = await filler.predict(`The capital of France is ${filler.maskToken}.`, { topK: 3 });

slots[0]?.position; // 7 — index in the encoded sequence
slots[0]?.predictions;
// [ { id: 5242,  token: " Paris",      score: 0.2530 },
//   { id: 39888, token: " Nice",       score: 0.1340 },
//   { id: 51596, token: " Strasbourg", score: 0.0429 } ]

await filler.dispose();
```

`topK` defaults to 5. Input with no mask token throws — a silent empty result would hide a typo in the
sentinel.

Two details that matter if you compare these numbers to anything:

- **Leading spaces are part of the token.** `" Paris"` and `"Paris"` are different ids with different
  scores; the model is picking a word *with* its preceding space.
- **Scores are true softmax probabilities over the full padded vocabulary.** The checkpoint pads to 65536
  for kernel alignment and only the first `realVocabSize` (64402) ids were ever trained. Untrained ids are
  excluded from the *candidates* but stay in the softmax denominator, which is what keeps these numbers
  comparable to the reference implementation's. They do not sum to 1 over the top-k.

Fill-mask is the task most exposed to quantization: `q8` puts `" Paris"` at 0.25 where fp32 says 0.44, and
reshuffles positions 2–5. Ordering above the top-2 is not stable at `q8`. Use `fp32` if you need the
distribution rather than the argmax.

| Type | Shape |
| --- | --- |
| `MaskPrediction` | `{ token: string; id: number; score: number }` |
| `MaskSlot` | `{ position: number; predictions: readonly MaskPrediction[] }` |
| `FillMask` | `{ model: EncoderModel; maskToken: string; predict(...); dispose() }` |

## Constants

| Constant | Value |
| --- | --- |
| `PROMPT_ROUTER_MODEL` | `"LFM2.5-Encoder-350M-Prompt-Router-ONNX"` |
| `POLICY_LINTER_MODEL` | `"LFM2.5-Encoder-350M-Policy-Linter-ONNX"` |
| `ENCODER_MODEL` | `"LFM2.5-Encoder-350M-ONNX"` |
| `DEFAULT_THRESHOLD` | `0.5` |

Repo ids are resolved relative to `modelRoot`, so these are the directory names `bun run export` writes.

## Lower-level: sharing a pass

`loadHead`, `runTwoTower` and `labelVectors` are the guts of the router and the linter. Use them when you
want the projections themselves — a custom pooling region, both heads' scores from one graph, an embedding
of a label.

```ts
loadHead(id: string, expected: "cosine" | "dot", options: LoadOptions): Promise<{ model; head }>
runTwoTower(model, head, text, labels): Promise<TwoTowerPass>
labelVectors(pass: TwoTowerPass): Float32Array[]
```

```ts
import { dot, normalize, poolSpan } from "@lfm-encoder/core";
import { PROMPT_ROUTER_MODEL, labelVectors, loadHead, runTwoTower } from "@lfm-encoder/tasks";

const { model, head } = await loadHead(PROMPT_ROUTER_MODEL, "cosine", { modelRoot: "./models" });

const pass = await runTwoTower(model, head, "my card was charged twice", ["billing", "sales"]);
// pass: { prefix, tokenized, tokenProj, ruleProj, labelSpans, textSpan }

const query = normalize(poolSpan(pass.tokenProj, pass.tokenized.spans, pass.textSpan));
const similarities = labelVectors(pass).map((label) => dot(normalize(label), query));
// [0.9994, -0.9970]  — raw cosines, before `* head.scale + head.bias` and the softmax

await model.dispose();
```

`loadHead` throws if the head kind does not match: pointing the router at the linter's weights produces
plausible-looking nonsense rather than an error, so the mismatch is caught at load time.

`tokenProj` is the query tower and `ruleProj` the key tower, both `(tokens, 256)`. The graph is identical
for the router and the linter; the heads differ only in the scalars in `head`
(`kind`, `scale`, `bias`, `activation`, `normalize`). See
[how the zero-shot heads work](../../README.md#how-the-zero-shot-heads-work) for why pooling these
per-token projections in JS is exact and not an approximation.
