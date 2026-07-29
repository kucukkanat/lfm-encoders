import { describe, expect, setDefaultTimeout, test } from "bun:test";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { AutoTokenizer, env, type PreTrainedTokenizer } from "@huggingface/transformers";
import { tokenizeWithSpans, tokensIn } from "@lfm-encoder/core";

// These tests load real ONNX graphs — up to 1.7 GB at fp32 — so a cold-cache
// session build alone can outlast bun's 5s default. Nothing here is mocked;
// the runtime is the thing under test.
setDefaultTimeout(120_000);

/**
 * The offset reconstruction against the real byte-level BPE. Nothing is mocked
 * and nothing is stubbed: a hand-written tokenizer would agree with the byte
 * arithmetic by construction and prove nothing.
 *
 * `router.test.ts` already checks the spans against HuggingFace's own
 * `return_offsets_mapping`. What is checked here is the property that has to
 * hold for text the fixtures do not contain — that the spans tile the input
 * exactly, and that each one really does address the characters its token came
 * from — across the scripts that break naive offset arithmetic: two-byte
 * accents, three-byte CJK, four-byte emoji that occupy two UTF-16 code units,
 * and a special token spelled out in the middle of ordinary prose.
 */
const MODEL_ROOT =
	process.env.LFM_MODEL_ROOT ??
	resolve(dirname(fileURLToPath(import.meta.url)), "../../../../models");
const MODEL = "kucukkanat/LFM2.5-Encoder-350M-Prompt-Router-ONNX";
const available = existsSync(join(MODEL_ROOT, MODEL, "tokenizer.json"));

const EMOJI = "ship it 🚀🎉 — the release 👩‍🚀 is live";
const SAMPLES: readonly (readonly [name: string, text: string])[] = [
	["ascii", "My invoice from last month charged me twice, can I get a refund?"],
	["german", "Grüße aus München — die Straße war größer, als wir dachten."],
	["japanese", "日本語のテキストです。トークナイザは正しく動きますか？"],
	["emoji", EMOJI],
	["a literal mask token", "Fill the <|mask|> in, please."],
	["an empty string", ""],
];

describe.if(available)("tokenizeWithSpans", () => {
	let loading: Promise<PreTrainedTokenizer> | undefined;
	const load = (): Promise<PreTrainedTokenizer> => {
		env.localModelPath = MODEL_ROOT;
		env.allowLocalModels = true;
		env.allowRemoteModels = false;
		loading ??= AutoTokenizer.from_pretrained(MODEL);
		return loading;
	};

	test.each(SAMPLES)("tiles %s with one span per id", async (_name, text) => {
		const { ids, spans } = tokenizeWithSpans(await load(), text);
		expect(spans).toHaveLength(ids.length);
		expect(spans.map((span) => text.slice(span.start, span.end)).join("")).toBe(text);
		expect(spans.at(-1)?.end).toBe(text.length);
		// Contiguous and non-decreasing: no character is covered twice and none
		// is skipped, which is what makes the join above a real round-trip
		// rather than a coincidence.
		spans.forEach((span, index) => {
			expect(span.start).toBe(index === 0 ? 0 : (spans[index - 1]?.end as number));
			expect(span.end).toBeGreaterThanOrEqual(span.start);
		});
	});

	test.each(SAMPLES.filter(([, text]) => text.length > 0))(
		"anchors each token of %s where the tokenizer's own decoder puts it",
		async (_name, text) => {
			const tokenizer = await load();
			const { ids, spans } = tokenizeWithSpans(tokenizer, text);
			// Tokens holding a partial UTF-8 sequence decode to U+FFFD and have
			// no characters of their own to compare against — they are exactly
			// the case that makes decode-and-accumulate the wrong way to build
			// offsets. Every token that does decode cleanly must land on its own
			// text, and there has to be at least one or this proves nothing.
			let compared = 0;
			ids.forEach((id, index) => {
				const span = spans[index] as { start: number; end: number };
				const decoded = tokenizer.decode([id]);
				if (span.start === span.end || decoded.includes("�")) return;
				compared++;
				expect(text.slice(span.start, span.end)).toBe(decoded);
			});
			expect(compared).toBeGreaterThan(0);
		},
	);

	test("never cuts an astral character in half", async () => {
		const { spans } = tokenizeWithSpans(await load(), EMOJI);
		// A boundary between a high and a low surrogate would slice a code point
		// into two lone halves — the failure mode of walking the string one
		// UTF-16 unit at a time while counting UTF-8 bytes.
		const isLeading = (index: number) => (EMOJI.charCodeAt(index) & 0xfc00) === 0xd800;
		const isTrailing = (index: number) => (EMOJI.charCodeAt(index) & 0xfc00) === 0xdc00;
		for (const boundary of spans.flatMap((span) => [span.start, span.end])) {
			if (boundary === 0 || boundary === EMOJI.length) continue;
			expect(isLeading(boundary - 1) && isTrailing(boundary)).toBe(false);
		}
	});

	test("keeps a literal <|mask|> written in the text as a single token", async () => {
		const tokenizer = await load();
		const text = "Fill the <|mask|> in, please.";
		const { ids, spans } = tokenizeWithSpans(tokenizer, text);
		const start = text.indexOf("<|mask|>");
		const covering = tokensIn(spans, { start, end: start + "<|mask|>".length });
		expect(covering).toHaveLength(1);
		expect(ids[covering[0] as number]).toBe(tokenizer.mask_token_id as number);
		expect(text.slice(spans[covering[0] as number]?.start, spans[covering[0] as number]?.end)).toBe(
			"<|mask|>",
		);
	});

	test("gives the inserted BOS a zero-width span so no pool ever picks it up", async () => {
		const tokenizer = await load();
		const text = "My invoice from last month charged me twice.";
		const { ids, spans } = tokenizeWithSpans(tokenizer, text);
		expect(ids[0]).toBe(tokenizer.bos_token_id as number);
		expect(spans[0]).toEqual({ start: 0, end: 0 });
		expect(tokensIn(spans, { start: 0, end: text.length })).not.toContain(0);
	});
});
