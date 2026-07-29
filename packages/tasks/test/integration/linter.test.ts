import { afterAll, describe, expect, setDefaultTimeout, test } from "bun:test";
import {
	DEFAULT_THRESHOLD,
	loadHead,
	loadPolicyLinter,
	POLICY_LINTER_MODEL,
	type PolicyLinter,
	PROMPT_ROUTER_MODEL,
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
 * The real dot-product head, against per-token probabilities captured from the
 * fp32 PyTorch model. Nothing is mocked.
 *
 * As in `router.test.ts`, the numeric parity check runs at fp32 — where the
 * whole pipeline is exact to ~1e-3 and drift means a bug — while q8 is only
 * held to the contract callers depend on. For the linter that contract is
 * coarser than the router's: q8 moves individual word probabilities by up to
 * ~0.5 (see `scripts/parity.ts`), so the assertion is that a blatant violation
 * still trips the threshold somewhere and that clean prose still does not.
 */
const available = hasModels(POLICY_LINTER_MODEL);
const cases = available ? fixtures<TwoTowerCase>(POLICY_LINTER_MODEL).cases : [];

const VIOLATION = "We guarantee a 300% return within six months, risk free.";
const CLEAN = "Our quarterly report will be published on the investor relations page.";
const RULES = ["no guarantees about financial returns", "no medical or health claims"];

/**
 * The fixture holds one reference row per *token* of the draft region, but
 * `lint()` reports one score per word — the strongest of its pieces. Regroup
 * the reference the same way so the two are comparable. This mirrors
 * `scripts/parity.ts`'s `wordScores()`; it lives here rather than being
 * imported because that file is an executable report, not a module.
 */
function wordScores(testCase: TwoTowerCase): number[][] {
	const reference = testCase.reference as number[][];
	const offset = testCase.prefix.length;
	const shown = testCase.offsets.filter((span) => span[1] > offset && span[0] !== span[1]);
	return [...testCase.text.matchAll(/\S+/gu)].map((match) => {
		const start = match.index + offset;
		const end = start + match[0].length;
		const rows = shown
			.map((span, row) => ({ span, row }))
			.filter(({ span }) => span[0] < end && span[1] > start)
			.map(({ row }) => reference[row] as number[]);
		return testCase.labels.map((_, rule) =>
			rows.length === 0 ? 0 : Math.max(...rows.map((row) => row[rule] as number)),
		);
	});
}

describe.if(available)("policy linter", () => {
	const linters = new Map<string, Promise<PolicyLinter>>();
	const load = (dtype: "fp32" | "q8"): Promise<PolicyLinter> => {
		const existing = linters.get(dtype);
		if (existing) return existing;
		const created = loadPolicyLinter({ modelRoot: MODEL_ROOT, dtype });
		linters.set(dtype, created);
		return created;
	};
	afterAll(async () => {
		for (const linter of linters.values()) await (await linter).dispose();
	});

	test.if(hasFp32(POLICY_LINTER_MODEL))("reproduces the PyTorch word scores at fp32", async () => {
		const linter = await load("fp32");
		for (const testCase of cases) {
			const result = await linter.lint(testCase.text, testCase.labels);
			const expected = wordScores(testCase);
			expect(result.words).toHaveLength(expected.length);
			result.words.forEach((word, index) => {
				expect(word.scores).toHaveLength(testCase.labels.length);
				expect(maxDelta(word.scores, expected[index] as number[])).toBeLessThan(1e-2);
			});
		}
	});

	test("anchors word spans in the original text, not the prefixed prompt", async () => {
		const linter = await load("q8");
		for (const testCase of cases) {
			const result = await linter.lint(testCase.text, testCase.labels);
			// Offsets that forgot to subtract the prefix would still be in range
			// for these inputs, so compare the sliced characters rather than the
			// numbers: only the correct offsets slice each word back out.
			expect(result.words.map((word) => testCase.text.slice(word.start, word.end))).toEqual(
				result.words.map((word) => word.text),
			);
			expect(result.words.map((word) => word.text)).toEqual(testCase.text.match(/\S+/gu) ?? []);
			expect(result.tokenCount).toBeGreaterThan(result.words.length);
		}
	});

	test("flags a blatant violation at q8", async () => {
		const linter = await load("q8");
		const result = await linter.lint(VIOLATION, RULES);
		const flags = result.flagged();

		expect(flags.length).toBeGreaterThan(0);
		expect(result.rules).toEqual(RULES);
		for (const flag of flags) {
			expect(flag.score).toBeGreaterThanOrEqual(DEFAULT_THRESHOLD);
			expect(flag.rule).toBe(RULES[flag.ruleIndex] as string);
			expect(VIOLATION.slice(flag.start, flag.end)).toBe(flag.text);
		}
		// The financial rule is the one being broken; the medical rule is in the
		// set precisely so an "everything scores high" regression is visible.
		expect(flags.every((flag) => flag.ruleIndex === 0)).toBe(true);
	});

	test("leaves a clean sentence alone at the default threshold", async () => {
		const linter = await load("q8");
		const result = await linter.lint(CLEAN, RULES);
		expect(result.words.length).toBeGreaterThan(0);
		expect(result.flagged()).toEqual([]);
		for (const word of result.words) {
			for (const score of word.scores) expect(score).toBeLessThan(DEFAULT_THRESHOLD);
		}
	});

	test("orders flags worst first and re-thresholds without re-inferring", async () => {
		const linter = await load("q8");
		const result = await linter.lint(VIOLATION, RULES);

		const flags = result.flagged(0.5);
		for (let i = 1; i < flags.length; i++) {
			expect(flags[i]?.score).toBeLessThanOrEqual(flags[i - 1]?.score as number);
		}
		expect(result.flagged()).toEqual(flags);
		// A lower bar can only add flags, and a bar above 1 admits nothing.
		expect(result.flagged(0.1).length).toBeGreaterThanOrEqual(flags.length);
		expect(result.flagged(0)).toHaveLength(result.words.length * RULES.length);
		expect(result.flagged(1.01)).toEqual([]);
	});

	test("rejects an empty rule set rather than guessing", async () => {
		const linter = await load("q8");
		await expect(linter.lint("anything", [])).rejects.toThrow(/at least one rule/);
	});

	test("rejects a head of the wrong kind instead of scoring nonsense", async () => {
		const { model } = await load("q8");
		// Sanity-check the guard's premise: the two heads really are distinguishable.
		expect(model.config.head?.kind).toBe("dot");
		await expect(
			loadHead(PROMPT_ROUTER_MODEL, "dot", { modelRoot: MODEL_ROOT, dtype: "q8" }),
		).rejects.toThrow(/has a 'cosine' head, expected 'dot'/);
	});
});
