/**
 * Measure what each quantization costs, from JavaScript.
 *
 *     bun run --filter @lfm-encoder/tasks parity
 *
 * `tools/export`'s own check answers "is the exported graph faithful?" using
 * Python's onnxruntime. This answers the question that decides what to ship:
 * how far the *JavaScript* runtime lands from the fp32 PyTorch reference. The
 * two disagree — notably on fp16 — because Node and Python link different
 * onnxruntime builds with different kernels, so the numbers a browser sees have
 * to be measured where a browser's runtime lives.
 */
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Dtype } from "@lfm-encoder/core";
import {
	ENCODER_MODEL,
	loadFillMask,
	loadPolicyLinter,
	loadPromptRouter,
	POLICY_LINTER_MODEL,
	PROMPT_ROUTER_MODEL,
} from "@lfm-encoder/tasks";

const MODEL_ROOT =
	process.env.LFM_MODEL_ROOT ?? resolve(dirname(fileURLToPath(import.meta.url)), "../../../models");

const DTYPES: Dtype[] = ["fp32", "q8", "q4"];

interface Row {
	model: string;
	dtype: Dtype;
	max: number;
	mean: number;
	disagreements: number;
	metric: string;
}

function load<T>(model: string): { cases: T[] } | undefined {
	const path = join(MODEL_ROOT, model, "fixtures.json");
	return existsSync(path) ? (JSON.parse(readFileSync(path, "utf8")) as { cases: T[] }) : undefined;
}

function summarize(
	model: string,
	dtype: Dtype,
	deltas: number[],
	disagreements: number,
	metric: string,
): Row {
	return {
		model,
		dtype,
		max: Math.max(...deltas),
		mean: deltas.reduce((a, b) => a + b, 0) / deltas.length,
		disagreements,
		metric,
	};
}

interface RouteCase {
	text: string;
	labels: string[];
	reference: number[];
}
interface LintCase {
	text: string;
	labels: string[];
	prefix: string;
	offsets: [number, number][];
	reference: number[][];
}
interface MaskCase {
	text: string;
	top_k: { id: number; prob: number }[];
}

async function routerRows(dtype: Dtype): Promise<Row | undefined> {
	const data = load<RouteCase>(PROMPT_ROUTER_MODEL);
	if (!data) return undefined;
	const router = await loadPromptRouter({ modelRoot: MODEL_ROOT, dtype });
	const deltas: number[] = [];
	let flips = 0;
	for (const testCase of data.cases) {
		const result = await router.route(testCase.text, testCase.labels);
		const byLabel = new Map(result.scores.map((s) => [s.label, s.score]));
		const got = testCase.labels.map((l) => byLabel.get(l) as number);
		deltas.push(Math.max(...got.map((g, i) => Math.abs(g - (testCase.reference[i] as number)))));
		const best = testCase.reference.indexOf(Math.max(...testCase.reference));
		if (testCase.labels.indexOf(result.top.label) !== best) flips++;
	}
	await router.dispose();
	return summarize("Prompt-Router", dtype, deltas, flips, "top-1");
}

async function linterRows(dtype: Dtype): Promise<Row | undefined> {
	const data = load<LintCase>(POLICY_LINTER_MODEL);
	if (!data) return undefined;
	const linter = await loadPolicyLinter({ modelRoot: MODEL_ROOT, dtype });
	const deltas: number[] = [];
	let flips = 0;
	for (const testCase of data.cases) {
		const result = await linter.lint(testCase.text, testCase.labels);
		// The fixture holds one row per *token* in the draft region; words are the
		// max over their pieces, so compare word scores against the same max.
		const expected = wordScores(testCase);
		let worst = 0;
		result.words.forEach((word, index) => {
			word.scores.forEach((score, rule) => {
				const want = expected[index]?.[rule];
				if (want === undefined) return;
				worst = Math.max(worst, Math.abs(score - want));
				if (score >= 0.5 !== want >= 0.5) flips++;
			});
		});
		deltas.push(worst);
	}
	await linter.dispose();
	return summarize("Policy-Linter", dtype, deltas, flips, "threshold");
}

/** Reference token scores regrouped into words, mirroring `lint()`'s grouping. */
function wordScores(testCase: LintCase): number[][] {
	const offset = testCase.prefix.length;
	const shown = testCase.offsets
		.map((span, index) => ({ span, index }))
		.filter(({ span }) => span[1] > offset && span[0] !== span[1]);
	return [...testCase.text.matchAll(/\S+/gu)].map((match) => {
		const start = match.index + offset;
		const end = start + match[0].length;
		const rows = shown
			.map((entry, row) => ({ entry, row }))
			.filter(({ entry }) => entry.span[0] < end && entry.span[1] > start)
			.map(({ row }) => testCase.reference[row] as number[]);
		const rules = testCase.labels.length;
		return Array.from({ length: rules }, (_, rule) =>
			rows.length === 0 ? 0 : Math.max(...rows.map((r) => r[rule] as number)),
		);
	});
}

async function maskRows(dtype: Dtype): Promise<Row | undefined> {
	const data = load<MaskCase>(ENCODER_MODEL);
	if (!data) return undefined;
	const filler = await loadFillMask({ modelRoot: MODEL_ROOT, dtype });
	const deltas: number[] = [];
	let missing = 0;
	for (const testCase of data.cases) {
		const [slot] = await filler.predict(testCase.text, { topK: 5 });
		const got = new Map(slot?.predictions.map((p) => [p.id, p.score]) ?? []);
		deltas.push(Math.max(...testCase.top_k.map((p) => Math.abs((got.get(p.id) ?? 0) - p.prob))));
		missing += testCase.top_k.filter((p) => !got.has(p.id)).length;
	}
	await filler.dispose();
	return summarize("Encoder (fill-mask)", dtype, deltas, missing, "top-5");
}

const rows: Row[] = [];
for (const dtype of DTYPES) {
	for (const runner of [routerRows, linterRows, maskRows]) {
		const row = await runner(dtype);
		if (row) rows.push(row);
	}
}

if (rows.length === 0) {
	console.error(`No fixtures under ${MODEL_ROOT}. Run: bun run export && bun run export:check`);
	process.exit(1);
}

const width = Math.max(...rows.map((r) => r.model.length));
console.log(`\nJavaScript runtime vs fp32 PyTorch reference  (models: ${MODEL_ROOT})\n`);
console.log(
	`${"model".padEnd(width)}  ${"dtype".padEnd(6)} ${"max Δ".padStart(9)} ${"mean Δ".padStart(9)}  disagreements`,
);
console.log("-".repeat(width + 46));
for (const row of rows.sort((a, b) => a.model.localeCompare(b.model))) {
	const format = (x: number) => (x < 1e-3 ? x.toExponential(1) : x.toFixed(4));
	console.log(
		`${row.model.padEnd(width)}  ${row.dtype.padEnd(6)} ${format(row.max).padStart(9)} ${format(row.mean).padStart(9)}  ${row.disagreements} (${row.metric})`,
	);
}
console.log("\nΔ is the largest absolute difference in a final probability.");
