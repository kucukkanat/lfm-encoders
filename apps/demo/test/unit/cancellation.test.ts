import { describe, expect, test } from "bun:test";
import { CANCELLED, createRegistry } from "../../src/cancellation.js";

/** A promise that never settles, standing in for a load nobody is going to wait out. */
const never = <T>(): Promise<T> => new Promise<T>(() => {});

/** A promise that settles after `ms`, standing in for work that beats the Stop. */
const after = <T>(ms: number, value: T): Promise<T> =>
	new Promise((resolve) => setTimeout(() => resolve(value), ms));

describe("createRegistry", () => {
	test("reports a request as running until it is abandoned", () => {
		const registry = createRegistry();
		expect(registry.aborted(1)).toBe(false);
		registry.abandon(1);
		expect(registry.aborted(1)).toBe(true);
	});

	test("tracks ids independently", () => {
		const registry = createRegistry();
		registry.abandon(1);
		expect(registry.aborted(2)).toBe(false);
	});

	test("resolves work that finishes before any Stop", async () => {
		const registry = createRegistry();
		expect(await registry.until(1, after(1, "scores"))).toBe("scores");
	});

	test("gives up on work parked indefinitely once the id is abandoned", async () => {
		const registry = createRegistry();
		const parked = registry.until(1, never<string>());
		registry.abandon(1);
		expect(await parked).toBe(CANCELLED);
	});

	test("gives up immediately when the id was abandoned before the wait began", async () => {
		const registry = createRegistry();
		registry.abandon(1);
		expect(await registry.until(1, never<string>())).toBe(CANCELLED);
	});

	test("only cuts short the id that was abandoned", async () => {
		const registry = createRegistry();
		const other = registry.until(2, after(1, "scores"));
		registry.abandon(1);
		expect(await other).toBe("scores");
	});

	test("propagates a rejection rather than swallowing it as a cancellation", async () => {
		const registry = createRegistry();
		const failing = registry.until(1, Promise.reject(new Error("bad_alloc")));
		await expect(failing).rejects.toThrow("bad_alloc");
	});

	test("a retired id does not pass its cancellation to the next request reusing it", async () => {
		const registry = createRegistry();
		registry.abandon(1);
		registry.retire(1);
		expect(registry.aborted(1)).toBe(false);
		expect(await registry.until(1, after(1, "scores"))).toBe("scores");
	});

	test("a Stop arriving after the id is retired does not affect a fresh wait", async () => {
		const registry = createRegistry();
		const stale = registry.until(1, never<string>());
		registry.retire(1);
		// The worker has moved on; this Stop names an id nobody is waiting on.
		registry.abandon(1);
		registry.retire(1);
		expect(await Promise.race([stale, after(5, "still parked")])).toBe("still parked");
	});
});
