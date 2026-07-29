import {
	buildPrefix,
	type EncoderModel,
	type HeadConfig,
	type LoadOptions,
	labelRanges,
	loadEncoderModel,
	type Matrix,
	poolSpan,
	type Span,
	type Tokenized,
	textSpan,
} from "@lfm-encoder/core";

/** One encoder pass over `prefix + text`, with both towers kept per token. */
export interface TwoTowerPass {
	readonly prefix: string;
	readonly tokenized: Tokenized;
	readonly tokenProj: Matrix;
	readonly ruleProj: Matrix;
	readonly labelSpans: readonly Span[];
	readonly textSpan: Span;
}

export async function runTwoTower(
	model: EncoderModel,
	head: HeadConfig,
	text: string,
	labels: readonly string[],
): Promise<TwoTowerPass> {
	const prefix = buildPrefix(head.prefixHeading, labels);
	const tokenized = model.tokenize(prefix + text);
	const outputs = await model.forward(tokenized);
	const tokenProj = outputs.token_proj;
	const ruleProj = outputs.rule_proj;
	if (tokenProj === undefined || ruleProj === undefined) {
		throw new Error(`model '${model.id}' is not a two-tower head (no token_proj/rule_proj)`);
	}
	return {
		prefix,
		tokenized,
		tokenProj,
		ruleProj,
		labelSpans: labelRanges(head.prefixHeading, labels),
		textSpan: textSpan(prefix, text),
	};
}

/** The 256-d vector for each label, mean-pooled over the tokens it spans. */
export function labelVectors(pass: TwoTowerPass): Float32Array[] {
	return pass.labelSpans.map((span) => poolSpan(pass.ruleProj, pass.tokenized.spans, span));
}

/**
 * Load a task head and assert it is the kind the caller expects.
 *
 * Pointing the router at the linter's weights produces plausible-looking
 * nonsense rather than an error, so the mismatch is caught at load time.
 */
export async function loadHead(
	id: string,
	expected: HeadConfig["kind"],
	options: LoadOptions,
): Promise<{ model: EncoderModel; head: HeadConfig }> {
	const model = await loadEncoderModel(id, options);
	const { head } = model.config;
	if (head === undefined) {
		throw new Error(`model '${id}' has no task head; expected a '${expected}' head`);
	}
	if (head.kind !== expected) {
		await model.dispose();
		throw new Error(`model '${id}' has a '${head.kind}' head, expected '${expected}'`);
	}
	return { model, head };
}
