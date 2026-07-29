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
	type Diffuser,
	type DiffusionFrame,
	type DiffusionResult,
	type FillMask,
	type LintResult,
	loadDiffuser,
	loadFillMask,
	loadPolicyLinter,
	loadPromptRouter,
	type PolicyLinter,
	type PromptRouter,
	type RouteResult,
} from "@lfm-encoder/tasks";

export type TaskName = "routing" | "linting" | "fill-mask" | "diffusion";

export interface Settings {
	readonly dtype: Dtype;
	readonly device: Device;
}

export type Request =
	| { id: number; kind: "preload"; task: TaskName; settings: Settings }
	| { id: number; kind: "route"; settings: Settings; text: string; labels: string[] }
	| { id: number; kind: "lint"; settings: Settings; text: string; rules: string[] }
	| { id: number; kind: "fill-mask"; settings: Settings; text: string; topK: number }
	| {
			id: number;
			kind: "diffuse";
			settings: Settings;
			prompt: string;
			maxNewTokens: number;
			steps: number;
			temperature: number;
	  }
	| { id: number; kind: "cancel" };

export type Response =
	| { id: number; status: "ok"; result: unknown; elapsedMs: number }
	| { id: number; status: "error"; message: string }
	| { id: number; status: "loading"; task: TaskName; file: string; fraction: number | undefined }
	| { id: number; status: "ready"; task: TaskName; loadMs: number }
	// Diffusion is the one task with something to show *during* the run: each
	// denoising pass is a frame, and the request only resolves at the end.
	| { id: number; status: "frame"; frame: DiffusionFrame };

type Loaded = PromptRouter | PolicyLinter | FillMask | Diffuser;

/**
 * Where weights come from.
 *
 * In dev the Vite middleware serves a local export off disk, which is fast to
 * iterate against. A deployed build has no such luxury — GitHub Pages will not
 * host 2.5 GB — so it omits `modelRoot` entirely and transformers.js fetches the
 * published repos from the Hugging Face Hub instead.
 */
const source = import.meta.env.DEV ? { modelRoot: "/models" } : {};

const loaders: Record<
	TaskName,
	(options: Parameters<typeof loadPromptRouter>[0]) => Promise<Loaded>
> = {
	routing: loadPromptRouter,
	linting: loadPolicyLinter,
	"fill-mask": loadFillMask,
	diffusion: loadDiffuser,
};

/**
 * Set while a generation is running, so a `cancel` request can stop it.
 *
 * Diffusion holds the queue for tens of seconds; without an out-of-band stop
 * the user cannot switch tabs or change precision until it finishes.
 */
let cancelled = false;

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
				...source,
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
		case "diffuse": {
			cancelled = false;
			const result: DiffusionResult = await (model as Diffuser).generate(request.prompt, {
				maxNewTokens: request.maxNewTokens,
				steps: request.steps,
				temperature: request.temperature,
				onFrame: (frame) => post({ id: request.id, status: "frame", frame }),
				get signal() {
					return { aborted: cancelled };
				},
			});
			return result;
		}
		case "cancel":
			return null;
	}
}

function taskFor(request: Request): TaskName {
	return request.kind === "preload"
		? request.task
		: request.kind === "diffuse"
			? "diffusion"
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
	// Cancellation must not be queued: the run it is meant to interrupt is what
	// is holding the queue.
	if (event.data.kind === "cancel") {
		cancelled = true;
		post({ id: event.data.id, status: "ok", result: null, elapsedMs: 0 });
		return;
	}
	// `respond` never rejects, so the chain cannot be poisoned by one bad request.
	queue = queue.then(() => respond(event.data));
});
