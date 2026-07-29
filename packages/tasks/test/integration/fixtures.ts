import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Where `bun run export` wrote the ONNX repos. These tests run the real graphs
 * against reference values captured from PyTorch — nothing here is mocked, so
 * without the weights there is nothing to test.
 */
export const MODEL_ROOT =
	process.env.LFM_MODEL_ROOT ??
	resolve(dirname(fileURLToPath(import.meta.url)), "../../../../models");

export interface TwoTowerCase {
	readonly text: string;
	readonly labels: string[];
	readonly prefix: string;
	readonly input_ids: number[];
	readonly offsets: [number, number][];
	readonly reference: number[] | number[][];
}

export interface MaskCase {
	readonly text: string;
	readonly input_ids: number[];
	readonly mask_position: number;
	readonly top_k: { id: number; token: string; prob: number }[];
}

export function fixtures<T>(model: string): { cases: T[] } {
	const path = join(MODEL_ROOT, model, "fixtures.json");
	if (!existsSync(path)) {
		throw new Error(
			`missing ${path}\nRun: bun run export && bun run export:check (see tools/export/README.md)`,
		);
	}
	return JSON.parse(readFileSync(path, "utf8")) as { cases: T[] };
}

export function hasModels(model: string): boolean {
	return existsSync(join(MODEL_ROOT, model, "fixtures.json"));
}

/**
 * The fp32 graph is only produced by `--keep-fp32`, so the strict numeric
 * parity tests opt in rather than failing on a default export.
 */
export function hasFp32(model: string): boolean {
	return existsSync(join(MODEL_ROOT, model, "onnx", "model.onnx"));
}

/** Largest absolute difference between two equal-length series. */
export function maxDelta(a: readonly number[], b: readonly number[]): number {
	return Math.max(...a.map((value, index) => Math.abs(value - (b[index] as number))));
}
