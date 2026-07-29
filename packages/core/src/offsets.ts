import type { PreTrainedTokenizer } from "@huggingface/transformers";
import type { Span, Tokenized } from "./types.js";

/**
 * Anchor every token back to the characters it came from.
 *
 * Both task heads pool over *character ranges* — "the tokens covering rule 2",
 * "the tokens after the prefix" — so without offsets there is no way to run
 * them. transformers.js does not expose `return_offsets_mapping`, so we
 * reconstruct the mapping ourselves.
 *
 * The reconstruction is exact rather than heuristic. This tokenizer is a plain
 * byte-level BPE: no normalizer, no byte fallback, so the token strings
 * concatenate back to the input losslessly, and in that alphabet every
 * character stands for exactly one UTF-8 byte. Token *byte* lengths are
 * therefore just string lengths, and a running sum gives byte offsets that we
 * translate into JavaScript string indices.
 *
 * The obvious alternative — `decode()` each id and accumulate lengths — is
 * wrong for exactly the languages this model is for: a token holding a partial
 * UTF-8 sequence decodes to U+FFFD, so offsets drift the moment text stops
 * being ASCII.
 */
export function tokenizeWithSpans(tokenizer: PreTrainedTokenizer, text: string): Tokenized {
	const encoded = tokenizer(text);
	const ids = Array.from(encoded.input_ids.data as ArrayLike<bigint | number>, Number);
	const pieces = tokenizer.tokenize(text);

	const byteToIndex = buildByteIndex(text);
	const byteLength = byteToIndex.length - 1;

	let cursor = 0;
	const pieceSpans: Span[] = pieces.map((piece) => {
		const start = cursor;
		cursor += piece.length;
		return { start, end: cursor };
	});
	if (cursor !== byteLength) {
		throw new Error(
			`token/text byte mismatch: pieces cover ${cursor} bytes, text is ${byteLength}. ` +
				"The tokenizer is not the byte-level BPE this mapping assumes.",
		);
	}

	const spans = pieceSpans.map(({ start, end }) => ({
		start: byteToIndex[start] as number,
		end: byteToIndex[end] as number,
	}));
	return { text, ids, spans: alignToIds(ids, spans, tokenizer) };
}

/**
 * `tokenize()` returns only the pieces of the text; `tokenizer()` may also
 * prepend a BOS. Give that inserted token a zero-width span so pooling — which
 * skips empty spans, exactly as the reference does — ignores it.
 */
function alignToIds(
	ids: readonly number[],
	spans: readonly Span[],
	tokenizer: PreTrainedTokenizer,
): readonly Span[] {
	if (ids.length === spans.length) return spans;
	const empty: Span = { start: 0, end: 0 };
	if (ids.length === spans.length + 1 && ids[0] === tokenizer.bos_token_id) {
		return [empty, ...spans];
	}
	throw new Error(
		`cannot align ${ids.length} ids to ${spans.length} token spans; ` +
			"the tokenizer adds special tokens this mapping does not model.",
	);
}

/**
 * For each UTF-8 byte offset, the JavaScript string index of the character
 * containing it. Length is `byteLength + 1` so an end offset is addressable.
 */
function buildByteIndex(text: string): Uint32Array {
	const bytes = new TextEncoder().encode(text).length;
	const index = new Uint32Array(bytes + 1);
	let byte = 0;
	for (let i = 0; i < text.length; ) {
		const codePoint = text.codePointAt(i) as number;
		const width = codePoint < 0x80 ? 1 : codePoint < 0x800 ? 2 : codePoint < 0x10000 ? 3 : 4;
		for (let k = 0; k < width; k++) index[byte + k] = i;
		byte += width;
		i += codePoint >= 0x10000 ? 2 : 1;
	}
	index[bytes] = text.length;
	return index;
}

/**
 * Indices of the tokens overlapping `span`.
 *
 * Zero-width tokens never match, which is what keeps BOS and other inserted
 * specials out of every pool.
 */
export function tokensIn(spans: readonly Span[], span: Span): number[] {
	const hits: number[] = [];
	for (let i = 0; i < spans.length; i++) {
		const token = spans[i] as Span;
		if (token.start < span.end && token.end > span.start && token.start !== token.end) {
			hits.push(i);
		}
	}
	return hits;
}
