import type {
	DiffusionFrame,
	DiffusionResult,
	LintWord,
	MaskSlot,
	RouteResult,
} from "@lfm-encoder/tasks";
import type { Request, Response, Settings, TaskName } from "./worker.js";

export interface LoadState {
	readonly task: TaskName;
	readonly file: string;
	readonly fraction: number | undefined;
}

/**
 * `Omit` collapses a union into its common keys; distributing over the members
 * first keeps each request variant's own fields.
 */
type WithoutId<T> = T extends unknown ? Omit<T, "id"> : never;

export interface LintData {
	readonly rules: readonly string[];
	readonly words: readonly LintWord[];
	readonly tokenCount: number;
}

/**
 * A request the UI can still walk away from.
 *
 * `cancel()` is advisory — see the worker for what it can and cannot interrupt
 * — so the outcome is carried by `promise`: a result if the run produced one,
 * `undefined` if it was dropped first.
 */
export interface Cancellable<T> {
	readonly promise: Promise<T | undefined>;
	cancel(): void;
}

/**
 * Promise-shaped wrapper around the worker's message protocol.
 *
 * Requests carry an incrementing id so several can be in flight without their
 * replies being confused — which matters because the UI fires on every
 * keystroke and a slow earlier request must not overwrite a newer result.
 */
export class InferenceClient {
	readonly #worker: Worker;
	readonly #pending = new Map<
		number,
		{ resolve: (v: unknown) => void; reject: (e: Error) => void }
	>();
	#nextId = 1;

	onLoading: ((state: LoadState) => void) | undefined;
	onReady: ((task: TaskName, loadMs: number) => void) | undefined;
	/** Wall-clock cost of the last completed forward pass. */
	onTiming: ((elapsedMs: number) => void) | undefined;
	/** One denoising pass finished; only masked diffusion emits these. */
	onFrame: ((frame: DiffusionFrame) => void) | undefined;

	constructor() {
		this.#worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
		this.#worker.addEventListener("message", (event: MessageEvent<Response>) => {
			const message = event.data;
			if (message.status === "loading") {
				this.onLoading?.({ task: message.task, file: message.file, fraction: message.fraction });
				return;
			}
			if (message.status === "ready") {
				this.onReady?.(message.task, message.loadMs);
				return;
			}
			if (message.status === "frame") {
				this.onFrame?.(message.frame);
				return;
			}
			const entry = this.#pending.get(message.id);
			if (!entry) return;
			this.#pending.delete(message.id);
			if (message.status === "ok") {
				this.onTiming?.(message.elapsedMs);
				entry.resolve(message.result);
			} else if (message.status === "cancelled") {
				entry.resolve(undefined);
			} else entry.reject(new Error(message.message));
		});
	}

	#post<T>(id: number, request: WithoutId<Request>): Promise<T | undefined> {
		return new Promise<T | undefined>((resolve, reject) => {
			this.#pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
			this.#worker.postMessage({ ...request, id } as Request);
		});
	}

	#send<T>(request: WithoutId<Request>): Promise<T | undefined> {
		return this.#post<T>(this.#nextId++, request);
	}

	#start<T>(request: WithoutId<Request>): Cancellable<T> {
		const id = this.#nextId++;
		return {
			promise: this.#post<T>(id, request),
			cancel: () => void this.#send({ kind: "cancel", target: id }),
		};
	}

	preload(task: TaskName, settings: Settings): Promise<void> {
		return this.#send<void>({ kind: "preload", task, settings });
	}

	route(settings: Settings, text: string, labels: string[]): Cancellable<RouteResult> {
		return this.#start({ kind: "route", settings, text, labels });
	}

	/** Heavy enough to be worth abandoning: one pass over the draft times every rule. */
	lint(settings: Settings, text: string, rules: string[]): Cancellable<LintData> {
		return this.#start({ kind: "lint", settings, text, rules });
	}

	fillMask(settings: Settings, text: string, topK: number): Cancellable<MaskSlot[]> {
		return this.#start({ kind: "fill-mask", settings, text, topK });
	}

	diffuse(
		settings: Settings,
		prompt: string,
		options: { maxNewTokens: number; steps: number; temperature: number },
	): Cancellable<DiffusionResult> {
		return this.#start({ kind: "diffuse", settings, prompt, ...options });
	}
}

/**
 * Run `start`, abandoning whatever the previous call left in flight.
 *
 * Panels re-run on input, so a typed sentence issues a request per pause. The
 * worker runs them strictly one at a time, and only the last answer is wanted —
 * so the earlier ones are pure latency in front of it. Discarding just the
 * *results* is not enough: on a backend where a pass costs seconds rather than
 * milliseconds, four superseded passes are half a minute during which the pane
 * still shows the answer to a sentence the user has finished editing, which
 * reads as the demo being broken rather than busy.
 *
 * Cancelling instead lets the worker drop them before they are dequeued, so the
 * newest request starts as soon as the one pass already running finishes.
 */
export function newest<A extends unknown[], R>(
	start: (...args: A) => Cancellable<R>,
): (...args: A) => Promise<R | undefined> {
	let current: Cancellable<R> | undefined;
	return async (...args: A) => {
		current?.cancel();
		const job = start(...args);
		current = job;
		const result = await job.promise;
		// A pass already under way cannot be interrupted, so a superseded request
		// can still come back with an answer. It is stale by definition.
		return job === current ? result : undefined;
	};
}
