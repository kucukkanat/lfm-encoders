import {
	dot,
	type EncoderModel,
	type LoadOptions,
	type Span,
	sigmoid,
	tokensIn,
} from "@lfm-encoder/core";
import { labelVectors, loadHead, runTwoTower } from "./two-tower.js";

export const POLICY_LINTER_MODEL = "LFM2.5-Encoder-350M-Policy-Linter-ONNX";

/** A word of the input, scored independently against every rule. */
export interface LintWord extends Span {
	readonly text: string;
	/** One probability per rule, in the order the rules were supplied. */
	readonly scores: readonly number[];
}

/** A word that crossed the threshold for a specific rule. */
export interface LintFlag extends Span {
	readonly text: string;
	readonly ruleIndex: number;
	readonly rule: string;
	readonly score: number;
}

export interface LintResult {
	readonly rules: readonly string[];
	/** Every word of the input, in order, whether or not it was flagged. */
	readonly words: readonly LintWord[];
	readonly tokenCount: number;
	/** Words above `threshold`, worst first. Re-runnable without re-inferring. */
	flagged(threshold?: number): LintFlag[];
}

export interface PolicyLinter {
	readonly model: EncoderModel;
	/**
	 * Score every word of `text` against every free-text `rule`, in one pass.
	 * Rules are prose and can change per call.
	 */
	lint(text: string, rules: readonly string[]): Promise<LintResult>;
	dispose(): Promise<void>;
}

export const DEFAULT_THRESHOLD = 0.5;

export async function loadPolicyLinter(options: LoadOptions = {}): Promise<PolicyLinter> {
	const { model, head } = await loadHead(POLICY_LINTER_MODEL, "dot", options);

	return {
		model,
		lint: async (text, rules) => {
			if (rules.length === 0) throw new Error("lint() needs at least one rule");
			const pass = await runTwoTower(model, head, text, rules);
			const ruleVectors = labelVectors(pass);

			// Words, not tokens: a byte-level BPE splits "guarantee" into pieces
			// that each carry their own score, and highlighting half a word reads
			// as a bug. Take the strongest piece, which is what the reference UI
			// does. Offsets are relative to `text`, so the prefix is subtracted.
			const offset = pass.prefix.length;
			const words = [...text.matchAll(/\S+/gu)].map((match) => {
				const start = match.index;
				const end = start + match[0].length;
				const pieces = tokensIn(pass.tokenized.spans, {
					start: start + offset,
					end: end + offset,
				});
				const scores = ruleVectors.map((rule) => {
					let best = Number.NEGATIVE_INFINITY;
					for (const piece of pieces) {
						best = Math.max(best, dot(pass.tokenProj.row(piece), rule));
					}
					return pieces.length === 0 ? 0 : sigmoid(best * head.scale + head.bias);
				});
				return { text: match[0], start, end, scores };
			});

			return {
				rules: [...rules],
				words,
				tokenCount: pass.tokenized.ids.length,
				flagged(threshold = DEFAULT_THRESHOLD) {
					const hits: LintFlag[] = [];
					for (const word of words) {
						word.scores.forEach((score, ruleIndex) => {
							if (score >= threshold) {
								hits.push({
									text: word.text,
									start: word.start,
									end: word.end,
									ruleIndex,
									rule: rules[ruleIndex] as string,
									score,
								});
							}
						});
					}
					return hits.sort((a, b) => b.score - a.score);
				},
			};
		},
		dispose: () => model.dispose(),
	};
}
