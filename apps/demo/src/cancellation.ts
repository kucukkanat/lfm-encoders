/**
 * Bookkeeping for requests the UI is allowed to walk away from.
 *
 * Split out of the worker so the concurrent part — a Stop racing a load, an id
 * being reused after it is retired — can be tested on its own, without pulling
 * onnxruntime and half a gigabyte of weights into the test run.
 */

/** Stand-in for work that was dropped before it produced anything. */
export const CANCELLED: unique symbol = Symbol("cancelled");
export type Cancelled = typeof CANCELLED;

export interface Registry {
	/** Whether `id` has been abandoned and not yet retired. */
	aborted(id: number): boolean;
	abandon(id: number): void;
	/**
	 * `work`, resolving to `CANCELLED` the moment `id` is abandoned.
	 *
	 * `work` is not itself interrupted — nothing here can do that — it is merely
	 * stopped being waited on. For a model load that is the point: it carries on
	 * into the resident cache and the next run starts warm.
	 */
	until<T>(id: number, work: Promise<T>): Promise<T | Cancelled>;
	/** Forget `id`, so a later request that reuses the number starts clean. */
	retire(id: number): void;
}

export function createRegistry(): Registry {
	const abandoned = new Set<number>();
	const wakeups = new Map<number, () => void>();

	return {
		aborted: (id) => abandoned.has(id),
		abandon: (id) => {
			abandoned.add(id);
			wakeups.get(id)?.();
		},
		until: (id, work) =>
			// Already abandoned is not a special case: the wakeup fires synchronously
			// below, so the race settles on `CANCELLED` without waiting on `work`.
			Promise.race([
				work,
				new Promise<Cancelled>((resolve) => {
					const wake = () => resolve(CANCELLED);
					wakeups.set(id, wake);
					if (abandoned.has(id)) wake();
				}),
			]),
		retire: (id) => {
			abandoned.delete(id);
			wakeups.delete(id);
		},
	};
}
