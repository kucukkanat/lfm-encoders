import { tokensIn } from "./offsets.js";
import type { Matrix, Span } from "./types.js";

/**
 * Mean of the rows at `indices`. Returns zeros when nothing is selected, which
 * is the reference behaviour for a label that tokenizes to nothing.
 */
export function meanRows(matrix: Matrix, indices: readonly number[]): Float32Array {
	const pooled = new Float32Array(matrix.cols);
	if (indices.length === 0) return pooled;
	for (const index of indices) {
		const row = matrix.row(index);
		for (let c = 0; c < pooled.length; c++) {
			pooled[c] = (pooled[c] as number) + (row[c] as number);
		}
	}
	for (let c = 0; c < pooled.length; c++) {
		pooled[c] = (pooled[c] as number) / indices.length;
	}
	return pooled;
}

/**
 * Mean-pool the projections of the tokens covering `span`.
 *
 * The reference heads pool hidden states and *then* project. Pooling already
 * projected tokens gives the same vector, because both towers are affine and an
 * affine map commutes with a mean. Doing it in this order is what lets the ONNX
 * graph stay independent of the label set: it emits projections per token and
 * never has to know how many rules there are.
 */
export function poolSpan(projections: Matrix, spans: readonly Span[], span: Span): Float32Array {
	return meanRows(projections, tokensIn(spans, span));
}

export function normalize(vector: Float32Array): Float32Array {
	let sum = 0;
	for (const value of vector) sum += value * value;
	const scale = 1 / Math.max(Math.sqrt(sum), 1e-12);
	const out = new Float32Array(vector.length);
	for (let i = 0; i < vector.length; i++) out[i] = (vector[i] as number) * scale;
	return out;
}

export function dot(a: Float32Array, b: Float32Array): number {
	let sum = 0;
	for (let i = 0; i < a.length; i++) sum += (a[i] as number) * (b[i] as number);
	return sum;
}

/** Numerically stable softmax over a small array of logits. */
export function softmax(logits: readonly number[]): number[] {
	const max = Math.max(...logits);
	const exponentials = logits.map((value) => Math.exp(value - max));
	const total = exponentials.reduce((a, b) => a + b, 0);
	return exponentials.map((value) => value / total);
}

export function sigmoid(value: number): number {
	return 1 / (1 + Math.exp(-value));
}
