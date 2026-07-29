import { describe, expect, test } from "bun:test";
import { dot, type Matrix, meanRows, normalize, sigmoid, softmax } from "@lfm-encoder/core";

/** The shape `model.forward()` hands back, built by hand so no session is needed. */
function matrix(rows: readonly number[][]): Matrix {
	const cols = rows[0]?.length ?? 0;
	const data = Float32Array.from(rows.flat());
	return {
		rows: rows.length,
		cols,
		data,
		row: (index) => data.subarray(index * cols, (index + 1) * cols),
	};
}

describe("meanRows", () => {
	const projections = matrix([
		[1, 2, 3],
		[100, 100, 100],
		[5, 6, 7],
	]);

	test("averages exactly the rows it is given", () => {
		expect([...meanRows(projections, [0, 2])]).toEqual([3, 4, 5]);
	});

	test("does not depend on the order of the indices", () => {
		expect([...meanRows(projections, [2, 0])]).toEqual([...meanRows(projections, [0, 2])]);
	});

	test("returns a zero vector of the right width when nothing is selected", () => {
		// A label that tokenizes to nothing pools to zeros rather than NaN, which
		// is what the reference implementation does and what keeps a downstream
		// dot product finite.
		const pooled = meanRows(projections, []);
		expect(pooled).toHaveLength(projections.cols);
		expect([...pooled]).toEqual([0, 0, 0]);
	});
});

describe("normalize", () => {
	test("returns a unit vector pointing the same way", () => {
		const unit = normalize(Float32Array.from([3, 0, 4]));
		expect(dot(unit, unit)).toBeCloseTo(1, 6);
		const expected = [0.6, 0, 0.8];
		for (let i = 0; i < expected.length; i++) {
			expect(unit[i]).toBeCloseTo(expected[i] as number, 6);
		}
	});

	test("is scale invariant, which is what makes cosine scoring meaningful", () => {
		const small = normalize(Float32Array.from([1, -2, 3]));
		const large = normalize(Float32Array.from([1000, -2000, 3000]));
		for (let i = 0; i < small.length; i++) {
			expect(large[i]).toBeCloseTo(small[i] as number, 5);
		}
	});

	test("leaves a zero vector at zero instead of producing NaN", () => {
		// An empty pool is a legitimate input here (see meanRows), so the epsilon
		// floor in the divisor is load-bearing, not defensive noise.
		const zeros = normalize(new Float32Array(4));
		expect([...zeros]).toEqual([0, 0, 0, 0]);
	});
});

describe("dot", () => {
	test("sums the elementwise products", () => {
		expect(dot(Float32Array.from([1, 2, 3]), Float32Array.from([4, 5, 6]))).toBe(32);
	});

	test("is zero for orthogonal vectors", () => {
		expect(dot(Float32Array.from([1, 0]), Float32Array.from([0, 1]))).toBe(0);
	});
});

describe("softmax", () => {
	test("returns a distribution over the labels", () => {
		const scores = softmax([2, 1, 0.5, -3]);
		expect(scores.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
		for (const score of scores) expect(score).toBeGreaterThan(0);
	});

	test("preserves the ordering of the logits", () => {
		const logits = [0.5, 3, -2, 1];
		const scores = softmax(logits);
		const byLogit = [...logits.keys()].sort(
			(a, b) => (logits[b] as number) - (logits[a] as number),
		);
		const byScore = [...scores.keys()].sort(
			(a, b) => (scores[b] as number) - (scores[a] as number),
		);
		expect(byScore).toEqual(byLogit);
	});

	test("splits ties evenly", () => {
		expect(softmax([4, 4])).toEqual([0.5, 0.5]);
	});

	test("is shift invariant and finite on logits that would overflow exp()", () => {
		// The router's logits are temperature-scaled cosines and stay small, but
		// the max-subtraction is the only thing standing between a caller with
		// large logits and Infinity/Infinity = NaN.
		const huge = softmax([1000, 1001, 1002]);
		expect(huge.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
		const shifted = softmax([0, 1, 2]);
		for (let i = 0; i < shifted.length; i++) {
			expect(huge[i]).toBeCloseTo(shifted[i] as number, 12);
		}
	});
});

describe("sigmoid", () => {
	test("maps zero to one half", () => {
		expect(sigmoid(0)).toBe(0.5);
	});

	test("is symmetric about the origin", () => {
		for (const value of [0.25, 1, 7]) expect(sigmoid(-value)).toBeCloseTo(1 - sigmoid(value), 12);
	});

	test("stays strictly inside (0, 1), so a flag score is never 0 or 1", () => {
		expect(sigmoid(-30)).toBeGreaterThan(0);
		expect(sigmoid(30)).toBeLessThan(1);
	});

	test("is monotonic, so thresholding it is the same as thresholding the logit", () => {
		const logits = [-30, -3, -0.1, 0, 0.1, 3, 30];
		for (let i = 1; i < logits.length; i++) {
			expect(sigmoid(logits[i] as number)).toBeGreaterThan(sigmoid(logits[i - 1] as number));
		}
	});
});
