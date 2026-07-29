import { describe, expect, test } from "bun:test";
import { type Span, tokensIn } from "@lfm-encoder/core";

/**
 * The spans of `"<bos>The quick brown"` — a zero-width inserted token followed
 * by a contiguous tiling, which is the shape `tokenizeWithSpans` always
 * produces.
 */
const spans: readonly Span[] = [
	{ start: 0, end: 0 },
	{ start: 0, end: 3 },
	{ start: 3, end: 9 },
	{ start: 9, end: 15 },
	{ start: 15, end: 15 },
];

describe("tokensIn", () => {
	test("selects the tokens a range lands on", () => {
		expect(tokensIn(spans, { start: 3, end: 9 })).toEqual([2]);
	});

	test("selects every token a range touches, not only whole ones", () => {
		// Label and word ranges routinely cut a byte-level BPE token in half; a
		// half-covered token still carries the score for those characters.
		expect(tokensIn(spans, { start: 2, end: 4 })).toEqual([1, 2]);
	});

	test("treats both ends as half-open, so abutting ranges do not bleed", () => {
		expect(tokensIn(spans, { start: 0, end: 3 })).toEqual([1]);
		expect(tokensIn(spans, { start: 9, end: 15 })).toEqual([3]);
	});

	test("returns indices in order", () => {
		expect(tokensIn(spans, { start: 0, end: 15 })).toEqual([1, 2, 3]);
	});

	test("skips zero-width tokens even when the range covers them", () => {
		// This is what keeps the BOS out of every pool without anyone having to
		// special-case it downstream.
		expect(tokensIn(spans, { start: 0, end: 15 })).not.toContain(0);
		expect(tokensIn(spans, { start: 0, end: 15 })).not.toContain(4);
	});

	test("never matches a zero-width range", () => {
		// A label that contributed no characters must pool to nothing rather
		// than silently adopting whichever token happens to start there.
		expect(tokensIn(spans, { start: 3, end: 3 })).toEqual([]);
		expect(tokensIn(spans, { start: 0, end: 0 })).toEqual([]);
	});

	test("returns nothing for a range outside the text", () => {
		expect(tokensIn(spans, { start: 20, end: 30 })).toEqual([]);
	});

	test("returns nothing when there are no tokens", () => {
		expect(tokensIn([], { start: 0, end: 10 })).toEqual([]);
	});
});
