/**
 * Time a forward pass at each precision.
 *
 *     bun run --filter @lfm-encoder/tasks bench
 *
 * Size and accuracy are the obvious axes when picking a dtype; speed is the one
 * people assume. Quantization is not automatically faster — int4 weights have to
 * be unpacked before every matmul, so on some backends a smaller file runs
 * slower than the fp32 it came from. This measures it instead of guessing.
 *
 * Node links onnxruntime's CPU build, so these are CPU numbers. The browser's
 * WASM and WebGPU backends are different kernels again; the demo prints its own
 * timings in the status bar.
 */
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Dtype } from "@lfm-encoder/core";
import { loadPromptRouter, PROMPT_ROUTER_MODEL } from "@lfm-encoder/tasks";

const MODEL_ROOT =
	process.env.LFM_MODEL_ROOT ?? resolve(dirname(fileURLToPath(import.meta.url)), "../../../models");

const FILES: Record<Dtype, string> = {
	fp32: "model.onnx",
	fp16: "model_fp16.onnx",
	q8: "model_quantized.onnx",
	q4: "model_q4.onnx",
	q4f16: "model_q4f16.onnx",
};

const LABELS = ["billing and payments", "technical support", "account management", "sales enquiry"];
const SHORT = "My invoice was charged twice.";
const LONG = `${SHORT} ${"The customer has contacted support about this before and is asking for an itemised breakdown of every charge on the account since January. ".repeat(6)}`;

const WARMUP = 2;
const RUNS = 8;

function median(values: number[]): number {
	const sorted = [...values].sort((a, b) => a - b);
	const middle = Math.floor(sorted.length / 2);
	return sorted.length % 2 === 0
		? ((sorted[middle - 1] as number) + (sorted[middle] as number)) / 2
		: (sorted[middle] as number);
}

interface Row {
	dtype: Dtype;
	loadMs: number;
	shortMs: number;
	shortTokens: number;
	longMs: number;
	longTokens: number;
}

const rows: Row[] = [];
for (const dtype of ["fp32", "q8", "q4"] as Dtype[]) {
	if (!existsSync(join(MODEL_ROOT, PROMPT_ROUTER_MODEL, "onnx", FILES[dtype]))) {
		console.log(`skipping ${dtype}: not exported`);
		continue;
	}
	const started = performance.now();
	const router = await loadPromptRouter({ modelRoot: MODEL_ROOT, dtype });
	const loadMs = performance.now() - started;

	const time = async (text: string): Promise<[number, number]> => {
		for (let i = 0; i < WARMUP; i++) await router.route(text, LABELS);
		const samples: number[] = [];
		let tokens = 0;
		for (let i = 0; i < RUNS; i++) {
			const at = performance.now();
			const result = await router.route(text, LABELS);
			samples.push(performance.now() - at);
			tokens = result.tokenCount;
		}
		return [median(samples), tokens];
	};

	const [shortMs, shortTokens] = await time(SHORT);
	const [longMs, longTokens] = await time(LONG);
	rows.push({ dtype, loadMs, shortMs, shortTokens, longMs, longTokens });
	await router.dispose();
}

if (rows.length === 0) {
	console.error(`No models under ${MODEL_ROOT}. Run: bun run export`);
	process.exit(1);
}

const short = rows[0]?.shortTokens ?? 0;
const long = rows[0]?.longTokens ?? 0;
console.log(`\nPrompt router forward pass, onnxruntime CPU (median of ${RUNS})\n`);
console.log(
	`${"dtype".padEnd(6)} ${"load".padStart(8)} ${`${short} tok`.padStart(10)} ${`${long} tok`.padStart(10)}`,
);
console.log("-".repeat(38));
for (const row of rows) {
	console.log(
		`${row.dtype.padEnd(6)} ${`${(row.loadMs / 1000).toFixed(1)}s`.padStart(8)} ${`${row.shortMs.toFixed(0)}ms`.padStart(10)} ${`${row.longMs.toFixed(0)}ms`.padStart(10)}`,
	);
}
console.log("\nload = cold session creation, files already on local disk.");
