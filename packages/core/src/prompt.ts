import type { Span } from "./types.js";

/**
 * The single string both zero-shot heads are fed:
 *
 *     <Heading>:
 *     - label one
 *     - label two
 *
 *     Text:
 *     <the text>
 *
 * Labels and text go through the encoder together, in one bidirectional pass —
 * that shared pass is the whole trick. Every token attends to every label, so
 * adding a category costs a few more tokens rather than another forward.
 *
 * The exact layout is load-bearing. It is what the model was trained on, and
 * the character arithmetic below depends on it byte for byte.
 */
export function buildPrefix(heading: string, labels: readonly string[]): string {
	const body = labels.length > 0 ? labels.map((label) => `- ${label}`).join("\n") : "- (none)";
	return `${heading}:\n${body}\n\nText:\n`;
}

/** Character span of each label inside `buildPrefix(heading, labels)`. */
export function labelRanges(heading: string, labels: readonly string[]): Span[] {
	const ranges: Span[] = [];
	let position = `${heading}:\n`.length;
	for (const label of labels) {
		const start = position + 2; // past the "- " bullet
		const end = start + label.length;
		ranges.push({ start, end });
		position = end + 1; // past the newline
	}
	return ranges;
}

/** Character span of the text region inside `prefix + text`. */
export function textSpan(prefix: string, text: string): Span {
	return { start: prefix.length, end: prefix.length + text.length };
}
