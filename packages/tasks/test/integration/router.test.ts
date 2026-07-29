import { afterAll, describe, expect, setDefaultTimeout, test } from "bun:test";
import { tokenizeWithSpans } from "@lfm-encoder/core";
import {
	loadHead,
	loadPromptRouter,
	POLICY_LINTER_MODEL,
	PROMPT_ROUTER_MODEL,
	type PromptRouter,
} from "@lfm-encoder/tasks";
import {
	fixtures,
	hasFp32,
	hasModels,
	MODEL_ROOT,
	maxDelta,
	type TwoTowerCase,
} from "./fixtures.js";

// These tests load real ONNX graphs — up to 1.7 GB at fp32 — so a cold-cache
// session build alone can outlast bun's 5s default. Nothing here is mocked;
// the runtime is the thing under test.
setDefaultTimeout(120_000);

/**
 * The real ONNX graph, against probabilities captured from the fp32 PyTorch
 * model. Nothing here is mocked: this is the only check that the TypeScript
 * pooling, the character-offset reconstruction and the exported graph agree
 * with the reference implementation end to end.
 *
 * Numeric parity is asserted at fp32, where the whole pipeline is exact to
 * ~1e-5 and any drift means a genuine bug. The quantized graphs are held to the
 * contract callers actually depend on — that the winning label does not move —
 * because their probabilities legitimately wander (see `scripts/parity.ts`).
 */
const available = hasModels(PROMPT_ROUTER_MODEL);
const cases = available ? fixtures<TwoTowerCase>(PROMPT_ROUTER_MODEL).cases : [];

describe.if(available)("prompt router", () => {
	const routers = new Map<string, Promise<PromptRouter>>();
	const load = (dtype: "fp32" | "q8"): Promise<PromptRouter> => {
		const existing = routers.get(dtype);
		if (existing) return existing;
		const created = loadPromptRouter({ modelRoot: MODEL_ROOT, dtype });
		routers.set(dtype, created);
		return created;
	};
	afterAll(async () => {
		for (const router of routers.values()) await (await router).dispose();
	});

	test("tokenizes identically to the HF fast tokenizer", async () => {
		const { model } = await load("q8");
		for (const testCase of cases) {
			const tokenized = tokenizeWithSpans(model.tokenizer, testCase.prefix + testCase.text);
			expect(tokenized.ids).toEqual(testCase.input_ids);
		}
	});

	test("reconstructs the character offsets return_offsets_mapping would give", async () => {
		const { model } = await load("q8");
		for (const testCase of cases) {
			const tokenized = tokenizeWithSpans(model.tokenizer, testCase.prefix + testCase.text);
			expect(tokenized.spans.map((s) => [s.start, s.end])).toEqual(testCase.offsets);
		}
	});

	test.if(hasFp32(PROMPT_ROUTER_MODEL))(
		"reproduces the PyTorch probabilities at fp32",
		async () => {
			const router = await load("fp32");
			for (const testCase of cases) {
				const result = await router.route(testCase.text, testCase.labels);
				const byLabel = new Map(result.scores.map((s) => [s.label, s.score]));
				const ordered = testCase.labels.map((label) => byLabel.get(label) as number);
				expect(maxDelta(ordered, testCase.reference as number[])).toBeLessThan(1e-3);
			}
		},
	);

	test.each(cases.map((c) => [c.text.slice(0, 44), c] as const))(
		"picks the reference winner at q8: %s",
		async (_name, testCase) => {
			const router = await load("q8");
			const result = await router.route(testCase.text, testCase.labels);
			const reference = testCase.reference as number[];
			const expected = testCase.labels[reference.indexOf(Math.max(...reference))];
			expect(result.top.label).toBe(expected as string);
		},
	);

	test("returns a sorted probability distribution over the supplied labels", async () => {
		const router = await load("q8");
		const labels = ["billing", "technical support", "sales"];
		const result = await router.route("my card was charged twice", labels);

		expect(result.scores).toHaveLength(labels.length);
		expect(result.scores.reduce((sum, s) => sum + s.score, 0)).toBeCloseTo(1, 5);
		for (let i = 1; i < result.scores.length; i++) {
			expect(result.scores[i]?.score).toBeLessThanOrEqual(result.scores[i - 1]?.score as number);
		}
		expect(result.tokenCount).toBeGreaterThan(0);
	});

	test("re-scores the same text when the label set changes", async () => {
		const router = await load("q8");
		const text = "how do I reset my password";
		const a = await router.route(text, ["account access", "billing"]);
		const b = await router.route(text, ["cooking recipes", "billing"]);
		expect(a.top.label).toBe("account access");
		expect(b.top.label).toBe("billing");
	});

	test("rejects an empty label set rather than guessing", async () => {
		const router = await load("q8");
		await expect(router.route("anything", [])).rejects.toThrow(/at least one label/);
	});

	test("rejects a head of the wrong kind instead of scoring nonsense", async () => {
		const { model } = await load("q8");
		// Sanity-check the guard's premise: the two heads really are distinguishable.
		expect(model.config.head?.kind).toBe("cosine");
		await expect(
			loadHead(POLICY_LINTER_MODEL, "cosine", { modelRoot: MODEL_ROOT, dtype: "q8" }),
		).rejects.toThrow(/has a 'dot' head, expected 'cosine'/);
	});
});
