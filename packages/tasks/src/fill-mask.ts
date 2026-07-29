import { type EncoderModel, type LoadOptions, loadEncoderModel } from "@lfm-encoder/core";

export const ENCODER_MODEL = "LFM2.5-Encoder-350M-ONNX";

export interface MaskPrediction {
	readonly token: string;
	readonly id: number;
	readonly score: number;
}

export interface MaskSlot {
	/** Index of the masked token in the encoded sequence. */
	readonly position: number;
	readonly predictions: readonly MaskPrediction[];
}

export interface FillMask {
	readonly model: EncoderModel;
	/** The literal string to place where a prediction should go, e.g. `<|mask|>`. */
	readonly maskToken: string;
	/**
	 * Predict every masked position at once. Being bidirectional, each mask is
	 * conditioned on the whole sentence rather than only its left context.
	 */
	predict(text: string, options?: { topK?: number }): Promise<MaskSlot[]>;
	dispose(): Promise<void>;
}

export async function loadFillMask(options: LoadOptions = {}): Promise<FillMask> {
	const model = await loadEncoderModel(ENCODER_MODEL, options);
	const maskId = model.tokenizer.mask_token_id;
	const maskToken = model.tokenizer.mask_token;
	if (typeof maskId !== "number" || typeof maskToken !== "string") {
		throw new Error(`model '${ENCODER_MODEL}' has no mask token`);
	}

	return {
		model,
		maskToken,
		predict: async (text, { topK = 5 } = {}) => {
			const tokenized = model.tokenize(text);
			const positions = tokenized.ids.flatMap((id, index) => (id === maskId ? [index] : []));
			if (positions.length === 0) {
				throw new Error(`no ${maskToken} in the input`);
			}
			const outputs = await model.forward(tokenized);
			const logits = outputs.logits;
			if (logits === undefined) {
				throw new Error(`model '${ENCODER_MODEL}' produced no logits`);
			}

			// The checkpoint pads its vocabulary to 65536 for kernel alignment. Ids
			// past the tokenizer's real vocabulary were never trained and decode to
			// nothing, so they are excluded from the *candidates* — but they stay in
			// the softmax denominator, which is what makes these numbers comparable
			// to the reference implementation's.
			const limit = Math.min(model.config.realVocabSize || logits.cols, logits.cols);
			return positions.map((position) => {
				const row = logits.row(position);
				const total = expSum(row);
				const max = maxOf(row);
				const best = topIndices(row.subarray(0, limit), topK);
				return {
					position,
					predictions: best.map((id) => ({
						id,
						token: model.tokenizer.decode([id]),
						score: Math.exp((row[id] as number) - max) / total,
					})),
				};
			});
		},
		dispose: () => model.dispose(),
	};
}

function maxOf(values: Float32Array): number {
	let max = Number.NEGATIVE_INFINITY;
	for (const value of values) if (value > max) max = value;
	return max;
}

/** Shifted sum of exponentials — the softmax denominator over the full row. */
function expSum(values: Float32Array): number {
	const max = maxOf(values);
	let total = 0;
	for (const value of values) total += Math.exp(value - max);
	return total;
}

/** Indices of the `count` largest values, descending. */
function topIndices(values: Float32Array, count: number): number[] {
	const size = Math.min(count, values.length);
	const best: number[] = [];
	for (let i = 0; i < values.length; i++) {
		if (best.length < size) {
			best.push(i);
			if (best.length === size) best.sort((a, b) => (values[b] as number) - (values[a] as number));
			continue;
		}
		const worst = best[size - 1] as number;
		if ((values[i] as number) <= (values[worst] as number)) continue;
		best[size - 1] = i;
		for (let j = size - 1; j > 0; j--) {
			const above = best[j - 1] as number;
			const here = best[j] as number;
			if ((values[here] as number) <= (values[above] as number)) break;
			best[j - 1] = here;
			best[j] = above;
		}
	}
	return best;
}
