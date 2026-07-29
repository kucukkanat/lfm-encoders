import { afterAll, describe, expect, setDefaultTimeout, test } from "bun:test";
import { ENCODER_MODEL, type FillMask, loadFillMask } from "@lfm-encoder/tasks";
import { fixtures, hasFp32, hasModels, type MaskCase, MODEL_ROOT, maxDelta } from "./fixtures.js";

// These tests load real ONNX graphs — up to 1.7 GB at fp32 — so a cold-cache
// session build alone can outlast bun's 5s default. Nothing here is mocked;
// the runtime is the thing under test.
setDefaultTimeout(120_000);

/**
 * The plain encoder's logits head, against the top-k distribution captured
 * from the fp32 PyTorch model. Nothing is mocked.
 *
 * Fill-mask is the one task with an absolute answer rather than a comparative
 * one, so q8 is held to a real semantic claim — Paris stays in the top five for
 * the capital of France — while the exact ids and probabilities are checked at
 * fp32, where the graph is faithful to ~1e-4 (see `scripts/parity.ts`).
 */
const available = hasModels(ENCODER_MODEL);
const cases = available ? fixtures<MaskCase>(ENCODER_MODEL).cases : [];

describe.if(available)("fill mask", () => {
	const fillers = new Map<string, Promise<FillMask>>();
	const load = (dtype: "fp32" | "q8"): Promise<FillMask> => {
		const existing = fillers.get(dtype);
		if (existing) return existing;
		const created = loadFillMask({ modelRoot: MODEL_ROOT, dtype });
		fillers.set(dtype, created);
		return created;
	};
	afterAll(async () => {
		for (const filler of fillers.values()) await (await filler).dispose();
	});

	test("puts Paris in the top five for the capital of France at q8", async () => {
		const filler = await load("q8");
		const slots = await filler.predict(`The capital of France is ${filler.maskToken}.`);
		expect(slots).toHaveLength(1);
		const predictions = slots[0]?.predictions ?? [];
		expect(predictions).toHaveLength(5);
		// The tokenizer keeps the leading space as part of the token, so the
		// comparison is on the word rather than on the exact piece.
		expect(predictions.map((prediction) => prediction.token.trim())).toContain("Paris");
	});

	test("returns one slot per mask, in the order they appear", async () => {
		const filler = await load("q8");
		const text = `The capital of ${filler.maskToken} is ${filler.maskToken}.`;
		const slots = await filler.predict(text, { topK: 3 });

		const { ids } = filler.model.tokenize(text);
		const masked = ids.flatMap((id, index) =>
			id === filler.model.tokenizer.mask_token_id ? [index] : [],
		);
		expect(masked).toHaveLength(2);
		expect(slots.map((slot) => slot.position)).toEqual(masked);
		for (const slot of slots) expect(slot.predictions).toHaveLength(3);
	});

	test("ranks predictions by descending probability", async () => {
		const filler = await load("q8");
		const slots = await filler.predict(`Paris is the capital of ${filler.maskToken}.`, {
			topK: 10,
		});
		const predictions = slots[0]?.predictions ?? [];
		expect(predictions).toHaveLength(10);
		for (const prediction of predictions) {
			expect(prediction.score).toBeGreaterThan(0);
			expect(prediction.score).toBeLessThan(1);
			// The checkpoint pads its vocabulary to 65536 for kernel alignment;
			// those ids were never trained and must never be offered.
			expect(prediction.id).toBeLessThan(filler.model.config.realVocabSize);
		}
		for (let i = 1; i < predictions.length; i++) {
			expect(predictions[i]?.score).toBeLessThanOrEqual(predictions[i - 1]?.score as number);
		}
	});

	test("rejects text with nothing to fill", async () => {
		const filler = await load("q8");
		await expect(filler.predict("The capital of France is Paris.")).rejects.toThrow(
			/no <\|mask\|> in the input/,
		);
	});

	test.if(hasFp32(ENCODER_MODEL))("reproduces the PyTorch top-k at fp32", async () => {
		const filler = await load("fp32");
		for (const testCase of cases) {
			const slots = await filler.predict(testCase.text, { topK: testCase.top_k.length });
			expect(slots).toHaveLength(1);
			const slot = slots[0];
			expect(slot?.position).toBe(testCase.mask_position);
			expect(slot?.predictions.map((prediction) => prediction.id)).toEqual(
				testCase.top_k.map((prediction) => prediction.id),
			);
			expect(slot?.predictions.map((prediction) => prediction.token)).toEqual(
				testCase.top_k.map((prediction) => prediction.token),
			);
			expect(
				maxDelta(
					slot?.predictions.map((prediction) => prediction.score) ?? [],
					testCase.top_k.map((prediction) => prediction.prob),
				),
			).toBeLessThan(1e-3);
		}
	});
});
