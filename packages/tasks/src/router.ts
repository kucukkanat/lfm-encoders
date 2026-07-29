import {
	dot,
	type EncoderModel,
	type LoadOptions,
	normalize,
	poolSpan,
	softmax,
} from "@lfm-encoder/core";
import { labelVectors, loadHead, runTwoTower } from "./two-tower.js";

export const PROMPT_ROUTER_MODEL = "kucukkanat/LFM2.5-Encoder-350M-Prompt-Router-ONNX";

export interface RouteScore {
	readonly label: string;
	/** Softmax probability across the supplied labels; the set sums to 1. */
	readonly score: number;
}

export interface RouteResult {
	/** Highest-scoring label. */
	readonly top: RouteScore;
	/** Every label, highest score first. */
	readonly scores: readonly RouteScore[];
	/** Tokens in the combined categories+text pass, for cost accounting. */
	readonly tokenCount: number;
}

export interface PromptRouter {
	readonly model: EncoderModel;
	/**
	 * Score `text` against free-text `labels`. The labels are ordinary prose and
	 * can change on every call — nothing is trained or cached per label set.
	 */
	route(text: string, labels: readonly string[]): Promise<RouteResult>;
	dispose(): Promise<void>;
}

export async function loadPromptRouter(options: LoadOptions = {}): Promise<PromptRouter> {
	const { model, head } = await loadHead(PROMPT_ROUTER_MODEL, "cosine", options);

	return {
		model,
		route: async (text, labels) => {
			if (labels.length === 0) throw new Error("route() needs at least one label");
			const pass = await runTwoTower(model, head, text, labels);

			// Query tower: the text, pooled then L2-normalised. Cosine against each
			// normalised label vector, scaled by the learned temperature, then a
			// softmax across labels — so scores are comparative, not absolute.
			const query = normalize(poolSpan(pass.tokenProj, pass.tokenized.spans, pass.textSpan));
			const logits = labelVectors(pass).map(
				(vector) => dot(normalize(vector), query) * head.scale + head.bias,
			);

			const scores = softmax(logits)
				.map((score, index) => ({ label: labels[index] as string, score }))
				.sort((a, b) => b.score - a.score);
			return {
				top: scores[0] as RouteScore,
				scores,
				tokenCount: pass.tokenized.ids.length,
			};
		},
		dispose: () => model.dispose(),
	};
}
