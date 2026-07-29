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
			} else entry.reject(new Error(message.message));
		});
	}

	#send<T>(request: WithoutId<Request>): Promise<T> {
		const id = this.#nextId++;
		return new Promise<T>((resolve, reject) => {
			this.#pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
			this.#worker.postMessage({ ...request, id } as Request);
		});
	}

	preload(task: TaskName, settings: Settings): Promise<void> {
		return this.#send({ kind: "preload", task, settings });
	}

	route(settings: Settings, text: string, labels: string[]): Promise<RouteResult> {
		return this.#send({ kind: "route", settings, text, labels });
	}

	lint(settings: Settings, text: string, rules: string[]): Promise<LintData> {
		return this.#send({ kind: "lint", settings, text, rules });
	}

	fillMask(settings: Settings, text: string, topK: number): Promise<MaskSlot[]> {
		return this.#send({ kind: "fill-mask", settings, text, topK });
	}

	diffuse(
		settings: Settings,
		prompt: string,
		options: { maxNewTokens: number; steps: number; temperature: number },
	): Promise<DiffusionResult> {
		return this.#send({ kind: "diffuse", settings, prompt, ...options });
	}

	cancel(): Promise<null> {
		return this.#send({ kind: "cancel" });
	}
}

/**
 * Run `task`, but only deliver the result if it is still the newest call.
 *
 * Every panel re-runs on input, and inference takes long enough that replies
 * arrive out of order. Dropping stale ones is what stops the results pane
 * flickering back to an older answer.
 */
export function latest<A extends unknown[], R>(
	task: (...args: A) => Promise<R>,
): (...args: A) => Promise<R | undefined> {
	let generation = 0;
	return async (...args: A) => {
		const mine = ++generation;
		const result = await task(...args);
		return mine === generation ? result : undefined;
	};
}
