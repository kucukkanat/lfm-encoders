import { describe, expect, test } from "bun:test";
import { type Cancellable, newest } from "../../src/client.js";

/**
 * A request that only settles when the test says so, and records whether it was
 * abandoned — standing in for a forward pass the worker has not started yet.
 */
function stub<T>(): Cancellable<T> & {
	settle: (value: T | undefined) => void;
	cancelled: boolean;
} {
	let settle: (value: T | undefined) => void = () => {};
	const promise = new Promise<T | undefined>((resolve) => {
		settle = resolve;
	});
	const job = {
		promise,
		cancel: () => {
			job.cancelled = true;
		},
		cancelled: false,
		settle: (value: T | undefined) => settle(value),
	};
	return job;
}

describe("newest", () => {
	test("delivers the result of a call nothing supersedes", async () => {
		const job = stub<string>();
		const run = newest(() => job);
		const result = run();
		job.settle("scores");
		expect(await result).toBe("scores");
	});

	test("abandons the previous request as soon as a newer one is made", () => {
		const first = stub<string>();
		const second = stub<string>();
		const jobs = [first, second];
		const run = newest(() => jobs.shift() as Cancellable<string>);

		void run();
		expect(first.cancelled).toBe(false);
		void run();
		expect(first.cancelled).toBe(true);
		expect(second.cancelled).toBe(false);
	});

	test("discards an answer from a superseded request that finished anyway", async () => {
		// A pass already inside onnxruntime cannot be interrupted, so a cancelled
		// request can still come back with a real result. It is stale regardless.
		const first = stub<string>();
		const second = stub<string>();
		const jobs = [first, second];
		const run = newest(() => jobs.shift() as Cancellable<string>);

		const stale = run();
		const fresh = run();
		first.settle("old scores");
		second.settle("new scores");
		expect(await stale).toBeUndefined();
		expect(await fresh).toBe("new scores");
	});

	test("passes arguments through to the request it starts", async () => {
		const seen: unknown[][] = [];
		const run = newest((...args: unknown[]) => {
			seen.push(args);
			const job = stub<number>();
			job.settle(args.length);
			return job;
		});
		expect(await run("text", ["a", "b"])).toBe(2);
		expect(seen).toEqual([["text", ["a", "b"]]]);
	});
});
