import { describe, expect, test } from "bun:test";
import { buildPrefix, labelRanges, type Span, textSpan } from "@lfm-encoder/core";

/**
 * The prompt layout is not cosmetic: it is the string the heads were trained
 * on, and it is the coordinate system every span in this library is expressed
 * in. So there are two things worth pinning down — the exact characters, and
 * that the ranges still address them. The round-trip below is the real
 * invariant; the literal-string checks only exist to catch a layout change
 * that the round-trip would happily follow.
 */
describe("buildPrefix", () => {
	test("emits the bulleted list the router was trained on", () => {
		expect(buildPrefix("Categories", ["billing and payments", "sales enquiry"])).toBe(
			"Categories:\n- billing and payments\n- sales enquiry\n\nText:\n",
		);
	});

	test("emits the same layout under the linter's heading", () => {
		expect(buildPrefix("Policy", ["no medical claims"])).toBe(
			"Policy:\n- no medical claims\n\nText:\n",
		);
	});

	test("keeps a bullet in place when there are no labels", () => {
		// A prompt with no list at all is off-distribution, so the empty case
		// still emits one — and it must not be mistaken for a real label.
		const prefix = buildPrefix("Policy", []);
		expect(prefix).toBe("Policy:\n- (none)\n\nText:\n");
		expect(labelRanges("Policy", [])).toEqual([]);
	});
});

describe("labelRanges", () => {
	const cases: readonly (readonly [name: string, heading: string, labels: string[]])[] = [
		["one label", "Categories", ["billing"]],
		["several labels", "Categories", ["billing and payments", "technical support", "sales"]],
		["labels of very different lengths", "Policy", ["x", "no disparaging competitors by name"]],
		// Ranges are JavaScript string indices, so anything wider than one UTF-16
		// code unit is where naive byte arithmetic would drift.
		["accented labels", "Policy", ["keine Rückgabe", "größere Beträge"]],
		["labels outside the BMP", "Categories", ["🚀 launch", "日本語の問い合わせ"]],
	];

	test.each(cases)("slices %s back out of the prefix", (_name, heading, labels) => {
		const prefix = buildPrefix(heading, labels);
		const ranges = labelRanges(heading, labels);
		expect(ranges).toHaveLength(labels.length);
		ranges.forEach((range, index) => {
			expect(prefix.slice(range.start, range.end)).toBe(labels[index] as string);
		});
	});

	test("leaves the bullet and the newline outside the range", () => {
		const heading = "Policy";
		const labels = ["no guarantees", "no health claims"];
		const prefix = buildPrefix(heading, labels);
		const ranges = labelRanges(heading, labels);
		const first = ranges[0] as Span;
		const second = ranges[1] as Span;
		expect(prefix.slice(first.start - 2, first.start)).toBe("- ");
		expect(prefix[first.end]).toBe("\n");
		expect(first.end).toBeLessThan(second.start);
	});
});

describe("textSpan", () => {
	test("addresses the text half of the concatenated prompt", () => {
		const prefix = buildPrefix("Policy", ["no guarantees about financial returns"]);
		const text = "We guarantee a 300% return within six months.";
		const span = textSpan(prefix, text);
		expect((prefix + text).slice(span.start, span.end)).toBe(text);
	});

	test("never overlaps the labels it follows", () => {
		const heading = "Categories";
		const labels = ["billing", "sales"];
		const prefix = buildPrefix(heading, labels);
		const span = textSpan(prefix, "my card was charged twice");
		for (const range of labelRanges(heading, labels)) {
			expect(range.end).toBeLessThanOrEqual(span.start);
		}
	});

	test("is zero-width for an empty text", () => {
		const prefix = buildPrefix("Policy", ["anything"]);
		expect(textSpan(prefix, "")).toEqual({ start: prefix.length, end: prefix.length });
	});
});
