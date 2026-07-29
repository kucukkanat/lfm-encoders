/**
 * Inference host.
 *
 * A 350M-parameter forward pass under WASM occupies a thread for tens to
 * hundreds of milliseconds. On the main thread that is a visibly frozen page —
 * no caret, no scrolling — so every model lives here and the UI only ever
 * exchanges messages.
 */
import type { Device, Dtype } from "@lfm-encoder/core";
import {
	type FillMask,
	type LintResult,
	loadFillMask,
	loadPolicyLinter,
	loadPromptRouter,
	type PolicyLinter,
	type PromptRouter,
	type RouteResult,
} from "@lfm-encoder/tasks";

export type TaskName = "routing" | "linting" | "fill-mask";

export interface Settings {
	readonly dtype: Dtype;
	readonly device: Device;
}

export type Request =
	| { id: number; kind: "preload"; task: TaskName; settings: Settings }
	| { id: number; kind: "route"; settings: Settings; text: string; labels: string[] }
	| { id: number; kind: "lint"; settings: Settings; text: string; rules: string[] }
	| { id: number; kind: "fill-mask"; settings: Settings; text: string; topK: number };

export type Response =
	| { id: number; status: "ok"; result: unknown; elapsedMs: number }
	| { id: number; status: "error"; message: string }
	| { id: number; status: "loading"; task: TaskName; file: string; fraction: number | undefined }
	| { id: number; status: "ready"; task: TaskName; loadMs: number };

type Loaded = PromptRouter | PolicyLinter | FillMask;

const MODEL_ROOT = "/models";
const loaders: Record<
	TaskName,
	(options: Parameters<typeof loadPromptRouter>[0]) => Promise<Loaded>
> = {
	routing: loadPromptRouter,
	linting: loadPolicyLinter,
	"fill-mask": loadFillMask,
};

/**
 * Exactly one model stays resident.
 *
 * Keeping every (task, dtype, device) combination the user tried sounds like a
 * nicety and is actually a memory leak: each session is 0.4-1.7 GB, they are
 * never released, and a few flips of the precision selector take the tab past
 * onnxruntime's allocation ceiling. The failure is opaque — every subsequent
 * load dies with `std::bad_alloc` and the page looks broken until it is
 * reloaded. Only one model is ever in use, so the previous one is disposed
 * before the next is built. Re-selecting something costs a session rebuild, not
 * a download: the weights are still in the browser's HTTP cache.
 */
let resident: { key: string; model: Promise<Loaded> } | undefined;

function get(task: TaskName, settings: Settings, requestId: number): Promise<Loaded> {
	const key = `${task}:${settings.dtype}:${settings.device}`;
	if (resident?.key === key) return resident.model;

	const previous = resident;
	const started = performance.now();
	// Release first, then build: holding both at once is what runs the tab out
	// of memory, and awaiting disposal keeps the peak to a single model.
	const loading = Promise.resolve(previous?.model)
		.then((old) => old?.dispose())
		.catch(() => undefined)
		.then(() =>
			loaders[task]({
				modelRoot: MODEL_ROOT,
				dtype: settings.dtype,
				device: settings.device,
				onProgress: ({ file, fraction }) => {
					post({ id: requestId, status: "loading", task, file, fraction });
				},
			}),
		)
		.then((model) => {
			post({ id: requestId, status: "ready", task, loadMs: performance.now() - started });
			return model;
		});

	// A failed load must not stay resident, or every retry returns the failure.
	loading.catch(() => {
		if (resident?.key === key) resident = undefined;
	});
	resident = { key, model: loading };
	return loading;
}

function post(message: Response): void {
	self.postMessage(message);
}

async function handle(request: Request): Promise<unknown> {
	const model = await get(taskFor(request), request.settings, request.id);
	switch (request.kind) {
		case "preload":
			return null;
		case "route":
			return (model as PromptRouter).route(
				request.text,
				request.labels,
			) satisfies Promise<RouteResult>;
		case "lint": {
			const result: LintResult = await (model as PolicyLinter).lint(request.text, request.rules);
			// `flagged()` is a method and would not survive structured cloning, so
			// the UI gets plain data and applies its own threshold to `words`.
			return { rules: result.rules, words: result.words, tokenCount: result.tokenCount };
		}
		case "fill-mask":
			return (model as FillMask).predict(request.text, { topK: request.topK });
	}
}

function taskFor(request: Request): TaskName {
	return request.kind === "preload"
		? request.task
		: request.kind === "route"
			? "routing"
			: request.kind === "lint"
				? "linting"
				: "fill-mask";
}

/**
 * Requests run strictly one at a time.
 *
 * Without this, switching tab or precision mid-inference calls `get()`, which
 * disposes the resident model while a forward pass is still executing on it.
 * onnxruntime then fails the in-flight run with `invalid session id: <handle>`
 * — the session it was given no longer exists. Serialising means nothing is
 * ever running when `get()` decides to swap models, which is the invariant the
 * single-resident cache needs to be safe.
 *
 * Nothing is lost by queueing: there is one model and one thread, so concurrent
 * runs would contend anyway, and the UI already debounces input and discards
 * results from superseded requests.
 */
let queue: Promise<unknown> = Promise.resolve();

async function respond(request: Request): Promise<void> {
	const started = performance.now();
	try {
		const result = await handle(request);
		post({ id: request.id, status: "ok", result, elapsedMs: performance.now() - started });
	} catch (error: unknown) {
		post({
			id: request.id,
			status: "error",
			message: error instanceof Error ? error.message : String(error),
		});
	}
}

self.addEventListener("message", (event: MessageEvent<Request>) => {
	// `respond` never rejects, so the chain cannot be poisoned by one bad request.
	queue = queue.then(() => respond(event.data));
});
