import type { Device, Dtype } from "@lfm-encoder/core";
import type { DiffusionResult, LintWord, MaskSlot, RouteResult } from "@lfm-encoder/tasks";
import "./app.css";
import { type Cancellable, InferenceClient, type LintData, newest } from "./client.js";
import { chipList, debounce, el, examples, need, percent, replace, ruleColor } from "./dom.js";
import type { Settings, TaskName } from "./worker.js";

const client = new InferenceClient();

// ---------------------------------------------------------------- settings

const dtypeSelect = need<HTMLSelectElement>("dtype");
const deviceSelect = need<HTMLSelectElement>("device");

// A deployed build streams from the Hub, which only carries the quantized
// graphs. Offering fp32 there produces a 404 on a file that was never uploaded,
// so the option is dropped rather than left to fail at load time.
if (!import.meta.env.DEV) {
	for (const option of dtypeSelect.querySelectorAll("option[data-local-only]")) {
		option.remove();
	}
}
const settings = (): Settings => ({
	dtype: dtypeSelect.value as Dtype,
	device: deviceSelect.value as Device,
});

// ------------------------------------------------------------------ status

const statusText = need("status-text");
const statusBar = need("status");
const meter = need("meter");
const meterFill = need("meter-fill");

function status(message: string, kind: "info" | "error" = "info"): void {
	statusText.textContent = message;
	statusBar.classList.toggle("status--error", kind === "error");
}

client.onLoading = ({ task, fraction }) => {
	meter.hidden = fraction === undefined;
	if (fraction !== undefined) meterFill.style.width = percent(fraction);
	status(
		fraction === undefined
			? `Downloading ${task} model…`
			: `Downloading ${task} model — ${percent(fraction)}`,
	);
};
let readyNote = "";
client.onReady = (task, loadMs) => {
	meter.hidden = true;
	readyNote = `${task} model loaded in ${(loadMs / 1000).toFixed(1)}s`;
	status(`${readyNote} — cached for next time.${backendNote}`);
};
client.onTiming = (elapsedMs) => {
	status(`${readyNote} · forward pass ${elapsedMs.toFixed(0)}ms`);
};

/**
 * Announce that a pass has started.
 *
 * Panels keep the previous answer on screen until a new one arrives, which is
 * right — blanking the pane on every keystroke would be worse. But a pass costs
 * milliseconds on WebGPU and seconds on WASM, and over seconds an unchanged pane
 * reads as the demo having ignored the edit rather than being busy with it.
 */
function working(what: string): void {
	status(`${what}…`);
}

/**
 * Whether this browser will actually hand onnxruntime a GPU adapter.
 *
 * `device: "auto"` falls back to WASM without saying so, and the gap is not
 * subtle: on this machine a routing pass measured ~0.24s under WebGPU against
 * ~3s under WASM. Firefox has no WebGPU on macOS, which is the whole of why the
 * demo feels broken there, so it is worth stating rather than leaving the user
 * to conclude the model is stuck. `navigator.gpu` merely existing proves
 * nothing — Firefox exposes it on platforms that cannot produce an adapter.
 */
let backendNote = "";
const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
void Promise.resolve(gpu?.requestAdapter())
	.catch(() => undefined)
	.then((adapter) => {
		if (!adapter)
			backendNote =
				" This browser has no WebGPU, so passes run on WASM — expect seconds, not milliseconds.";
	});

function reportError(error: unknown): void {
	meter.hidden = true;
	const message = error instanceof Error ? error.message : String(error);
	const missing = /not found|404|could not locate/i.test(message);
	status(
		missing && import.meta.env.DEV
			? `Missing weights. Run \`bun run export\` from the repo root. (${message})`
			: missing
				? `That precision is not published for this model. (${message})`
				: message,
		"error",
	);
}

// -------------------------------------------------------------------- tabs

const panels = new Map<TaskName, HTMLElement>([
	["routing", need("panel-routing")],
	["linting", need("panel-linting")],
	["fill-mask", need("panel-fill-mask")],
	["diffusion", need("panel-diffusion")],
]);
const runners = new Map<TaskName, () => void>();
let active: TaskName = "routing";

/**
 * Wrap a panel's runner so it does nothing once its tab is no longer showing.
 *
 * Panels debounce, so a keystroke can fire its inference *after* the user has
 * moved on. `newest()` abandons the previous request, but nothing abandons one
 * whose panel has simply gone away, and the worker keeps exactly one model
 * resident — so a late routing pass evicts what the new tab just loaded and
 * forces a re-download.
 * That was survivable when every model was one forward pass; with a 424 MB
 * diffusion checkpoint it is not.
 */
function whileActive(task: TaskName, run: () => void): () => void {
	return () => {
		if (active === task) run();
	};
}

for (const tab of need("tabs").querySelectorAll<HTMLButtonElement>(".tab")) {
	tab.addEventListener("click", () => {
		active = tab.dataset.panel as TaskName;
		for (const other of need("tabs").querySelectorAll<HTMLButtonElement>(".tab")) {
			other.setAttribute("aria-selected", String(other === tab));
		}
		for (const [name, panel] of panels) panel.hidden = name !== active;
		restrictPrecision();
		runners.get(active)?.();
	});
}

/**
 * `q4` is not published for the diffusion checkpoint — it fails the exporter's
 * parity gate, because a loop that conditions each pass on the last cannot
 * absorb the drift a one-shot encoder can. Offering it here would produce a 404
 * on a file that was deliberately never uploaded, so the option is disabled
 * rather than left to fail at load time.
 */
function restrictPrecision(): void {
	const option = dtypeSelect.querySelector<HTMLOptionElement>('option[value="q4"]');
	if (!option) return;
	option.disabled = active === "diffusion";
	if (option.disabled && dtypeSelect.value === "q4") dtypeSelect.value = "q8";
}
restrictPrecision();

// Switching precision or backend invalidates whatever is on screen, so re-run.
for (const select of [dtypeSelect, deviceSelect]) {
	select.addEventListener("change", () => runners.get(active)?.());
}

// --------------------------------------------------------------- routing

function setupRouting(): void {
	const text = need<HTMLTextAreaElement>("route-text");
	const output = need("route-scores");
	const meta = need("route-meta");

	const SUPPORT = [
		"billing and payments",
		"technical support",
		"account management",
		"sales enquiry",
	];
	const labels = chipList(need("route-labels"), need<HTMLFormElement>("route-add"), SUPPORT, {
		onChange: () => run(),
	});

	const infer = newest((body: string, categories: string[]) =>
		client.route(settings(), body, categories),
	);

	const run = debounce(250, () => {
		if (active !== "routing") return;
		const body = text.value.trim();
		const categories = labels.items();
		if (body === "" || categories.length === 0) {
			replace(output, [el("li", "empty", "Add some text and at least one category.")]);
			return;
		}
		working("Routing");
		infer(body, categories)
			.then((result) => result && render(result))
			.catch(reportError);
	});

	function render(result: RouteResult): void {
		replace(
			output,
			result.scores.map((score, index) => {
				const item = el("li", index === 0 ? "score score--top" : "score");
				const row = el("div", "score__row");
				row.append(
					el("span", "score__label", score.label),
					el("span", "score__value", percent(score.score)),
				);
				const track = el("div", "score__track");
				const bar = el("div", "score__bar");
				bar.style.width = percent(score.score);
				track.append(bar);
				item.append(row, track);
				return item;
			}),
		);
		meta.textContent = `${result.tokenCount} tokens · ${result.scores.length} categories · one forward pass`;
	}

	examples(need("route-examples"), [
		{
			label: "German refund request",
			apply: () => {
				text.value = "Ich möchte mein Abonnement kündigen und eine Rückerstattung erhalten.";
				labels.set(SUPPORT);
			},
		},
		{
			label: "Crash report",
			apply: () => {
				text.value = "The deployment keeps crashing with an out-of-memory error on startup.";
				labels.set(SUPPORT);
			},
		},
		{
			label: "Route by intent instead",
			apply: () => {
				text.value = "Write a haiku about the ocean at dawn.";
				labels.set(["creative writing", "code generation", "factual question", "small talk"]);
			},
		},
	]);

	text.addEventListener("input", run);
	runners.set("routing", whileActive("routing", run));
}

// --------------------------------------------------------------- linting

function setupLinting(): void {
	const text = need<HTMLTextAreaElement>("lint-text");
	const output = need("lint-output");
	const legend = need("lint-legend");
	const meta = need("lint-meta");
	const threshold = need<HTMLInputElement>("lint-threshold");
	const thresholdValue = need<HTMLOutputElement>("lint-threshold-value");
	const runButton = need<HTMLButtonElement>("lint-run");
	const stopButton = need<HTMLButtonElement>("lint-stop");
	const staleNote = need("lint-stale");

	const POLICY = [
		"no guarantees about financial returns",
		"no medical or health claims",
		"no disparaging competitors by name",
	];
	const rules = chipList(need("lint-rules"), need<HTMLFormElement>("lint-add"), POLICY, {
		colored: true,
		onChange: () => touch(),
	});

	/**
	 * The last result, together with the draft it was computed from.
	 *
	 * Word spans are offsets into the exact string that was scored. While the
	 * panel re-ran on every keystroke that string was always the live textarea;
	 * now that scoring is manual it is not, and re-rendering against an edited
	 * textarea would underline the wrong characters. Keeping the source next to
	 * the scores lets the threshold slider keep working on the old result while
	 * the user types the next draft.
	 */
	let current: (LintData & { body: string }) | undefined;
	/** The run in flight, so Stop has something to aim at. */
	let job: Cancellable<LintData> | undefined;

	function run(): void {
		if (job) return;
		const body = text.value.trim();
		const policy = rules.items();
		if (body === "" || policy.length === 0) {
			replace(output, [el("span", "empty", "Add a draft and at least one rule.")]);
			replace(legend, []);
			return;
		}
		runButton.disabled = true;
		stopButton.disabled = false;
		const pending = client.lint(settings(), body, policy);
		job = pending;
		pending.promise
			.then((result) => {
				// `undefined` means Stop landed before the pass began; the previous
				// result stays on screen rather than being blanked for no reason.
				if (!result) {
					status("Lint cancelled.");
					return;
				}
				current = { ...result, body };
				render();
			})
			.catch(reportError)
			.finally(() => {
				job = undefined;
				runButton.disabled = false;
				stopButton.disabled = true;
			});
	}

	/** Whether what is on screen still describes the draft and policy as they now stand. */
	function stale(): boolean {
		return (
			current !== undefined &&
			(current.body !== text.value.trim() || current.rules.join("\n") !== rules.items().join("\n"))
		);
	}

	function touch(): void {
		staleNote.hidden = !stale();
		output.classList.toggle("highlighted--stale", stale());
	}

	/**
	 * Rebuild the highlighted text from word spans.
	 *
	 * The spans index the original string, so the gaps between them — the
	 * whitespace and punctuation the model never scored — are copied through
	 * verbatim. That keeps the output character-identical to the input.
	 */
	function render(): void {
		if (!current) return;
		const body = current.body;
		const limit = Number(threshold.value);
		const counts = current.rules.map(() => 0);
		const nodes: Node[] = [];
		let cursor = 0;

		for (const word of current.words) {
			const best = strongestRule(word, limit);
			if (best === undefined) continue;
			counts[best] = (counts[best] ?? 0) + 1;
			nodes.push(document.createTextNode(body.slice(cursor, word.start)));
			const mark = el("span", "hit", body.slice(word.start, word.end));
			mark.style.setProperty("--hit-color", ruleColor(best));
			mark.title = `${current.rules[best]} — ${percent(word.scores[best] as number)}`;
			nodes.push(mark);
			cursor = word.end;
		}
		nodes.push(document.createTextNode(body.slice(cursor)));
		replace(output, nodes);

		replace(
			legend,
			current.rules.map((rule, index) => {
				const item = el("li", "legend__item");
				const swatch = el("span", "legend__swatch");
				swatch.style.setProperty("--swatch", ruleColor(index));
				item.append(
					swatch,
					el("span", undefined, rule),
					el("span", "legend__count", `${counts[index] ?? 0}`),
				);
				return item;
			}),
		);
		const flagged = counts.reduce((a, b) => a + b, 0);
		meta.textContent = `${current.tokenCount} tokens · ${current.words.length} words · ${flagged} flagged at ≥ ${limit.toFixed(2)}`;
		touch();
	}

	/** Index of the highest-scoring rule for this word, if any clears `limit`. */
	function strongestRule(word: LintWord, limit: number): number | undefined {
		let best: number | undefined;
		let bestScore = limit;
		word.scores.forEach((score, index) => {
			if (score >= bestScore) {
				bestScore = score;
				best = index;
			}
		});
		return best;
	}

	threshold.addEventListener("input", () => {
		thresholdValue.textContent = Number(threshold.value).toFixed(2);
		// Re-thresholding is pure presentation: no second forward pass.
		render();
	});

	examples(need("lint-examples"), [
		{
			label: "Medical claim",
			apply: () => {
				text.value = "Unlike Acme Corp, our product actually cures chronic back pain.";
				rules.set(POLICY);
			},
		},
		{
			label: "Clean copy",
			apply: () => {
				text.value = "Our quarterly report will be published on the investor relations page.";
				rules.set(POLICY);
			},
		},
		{
			label: "French, financial guarantee",
			apply: () => {
				text.value = "Nous garantissons un rendement de 20 % par an, sans aucun risque.";
				rules.set(POLICY);
			},
		},
	]);

	runButton.addEventListener("click", run);
	stopButton.addEventListener("click", () => job?.cancel());
	text.addEventListener("input", touch);

	// Like diffusion, this panel never scores on its own. Switching to the tab
	// only warms the model, so the download happens while the user is reading the
	// blurb rather than after they ask for a result.
	runners.set("linting", () => {
		client.preload("linting", settings()).catch(reportError);
	});
}

// ------------------------------------------------------------- fill-mask

function setupFillMask(): void {
	const text = need<HTMLTextAreaElement>("mask-text");
	const output = need("mask-output");
	const meta = need("mask-meta");

	const infer = newest((body: string) => client.fillMask(settings(), body, 5));

	const run = debounce(400, () => {
		if (active !== "fill-mask") return;
		const body = text.value.trim();
		if (!body.includes("<|mask|>")) {
			replace(output, [el("p", "empty", "Put <|mask|> somewhere in the text.")]);
			return;
		}
		working("Predicting");
		infer(body)
			.then((slots) => slots && render(slots))
			.catch(reportError);
	});

	function render(slots: MaskSlot[]): void {
		replace(
			output,
			slots.map((slot, index) => {
				const section = el("div", "slot");
				if (slots.length > 1) {
					section.append(el("span", "slot__title", `Blank ${index + 1}`));
				}
				const list = el("ul", "predictions");
				for (const prediction of slot.predictions) {
					const item = el("li", "prediction");
					item.append(
						el("span", "prediction__token", JSON.stringify(prediction.token)),
						el("span", "prediction__score", percent(prediction.score)),
					);
					list.append(item);
				}
				section.append(list);
				return section;
			}),
		);
		meta.textContent = `${slots.length} blank${slots.length === 1 ? "" : "s"} predicted in one pass`;
	}

	examples(need("mask-examples"), [
		{ label: "Two blanks at once", apply: () => set("The <|mask|> of France is <|mask|>.") },
		{ label: "Spanish", apply: () => set("El <|mask|> es un animal doméstico.") },
		{ label: "Japanese", apply: () => set("日本の首都は<|mask|>です。") },
	]);

	function set(value: string): void {
		text.value = value;
		run();
	}

	text.addEventListener("input", run);
	runners.set("fill-mask", whileActive("fill-mask", run));
}

// ------------------------------------------------------------- diffusion

function setupDiffusion(): void {
	const text = need<HTMLTextAreaElement>("diffusion-text");
	const canvas = need("diffusion-canvas");
	const answer = need("diffusion-answer");
	const meta = need("diffusion-meta");
	const runButton = need<HTMLButtonElement>("diffusion-run");
	const stopButton = need<HTMLButtonElement>("diffusion-stop");

	const sliders = [
		["diffusion-length", "diffusion-length-value", (v: number) => String(v)],
		["diffusion-steps", "diffusion-steps-value", (v: number) => String(v)],
		["diffusion-temp", "diffusion-temp-value", (v: number) => v.toFixed(2)],
	] as const;
	for (const [input, output, format] of sliders) {
		const slider = need<HTMLInputElement>(input);
		const label = need<HTMLOutputElement>(output);
		slider.addEventListener("input", () => {
			label.textContent = format(Number(slider.value));
		});
	}

	let job: Cancellable<DiffusionResult> | undefined;
	let started = 0;

	/**
	 * Draw the canvas.
	 *
	 * Slots still masked are drawn as placeholders rather than omitted, so the
	 * answer's final length is visible from the first frame and tokens do not
	 * jump around as they land — the whole point of showing the process.
	 */
	client.onFrame = (frame) => {
		const fresh = new Set(frame.revealed);
		replace(
			canvas,
			frame.tokens.map((token, index) => {
				if (token === null) return el("span", "slot-mask", "▁");
				const node = el("span", fresh.has(index) ? "slot-token slot-token--new" : "slot-token");
				node.textContent = token;
				return node;
			}),
		);
		const filled = frame.tokens.filter((token) => token !== null).length;
		const elapsed = `${((performance.now() - started) / 1000).toFixed(1)}s`;
		meta.textContent = `pass ${frame.step} · ${filled}/${frame.tokens.length} slots committed · ${elapsed}`;
		// A run is tens of forward passes rather than one, so the shared status bar
		// would otherwise sit on whatever the last *load* said for the whole
		// generation and look stuck.
		status(`denoising — pass ${frame.step}, ${filled}/${frame.tokens.length} slots · ${elapsed}`);
	};

	function run(): void {
		if (job) return;
		const prompt = text.value.trim();
		if (prompt === "") {
			replace(canvas, [el("p", "empty", "Ask something first.")]);
			return;
		}
		started = performance.now();
		runButton.disabled = true;
		stopButton.disabled = false;
		replace(answer, []);
		const pending = client.diffuse(settings(), prompt, {
			maxNewTokens: Number(need<HTMLInputElement>("diffusion-length").value),
			steps: Number(need<HTMLInputElement>("diffusion-steps").value),
			temperature: Number(need<HTMLInputElement>("diffusion-temp").value),
		});
		job = pending;
		pending.promise
			.then((result) => {
				// Stopping mid-generation still yields the canvas as it stands; only a
				// Stop that lands before the first pass leaves nothing to show.
				if (!result) {
					status("Generation cancelled.");
					return;
				}
				replace(answer, [el("p", undefined, result.text || "(empty)")]);
				const elapsed = `${((performance.now() - started) / 1000).toFixed(1)}s`;
				meta.textContent = `${result.steps} passes · ${result.promptTokens} prompt tokens · ${result.canvasTokens} on the canvas · ${elapsed}`;
				status(`answered in ${result.steps} passes · ${elapsed}`);
			})
			.catch(reportError)
			.finally(() => {
				job = undefined;
				runButton.disabled = false;
				stopButton.disabled = true;
			});
	}

	runButton.addEventListener("click", run);
	stopButton.addEventListener("click", () => job?.cancel());

	examples(need("diffusion-examples"), [
		{ label: "Haiku", apply: () => set("Write a haiku about the ocean at dawn.") },
		{ label: "Arithmetic", apply: () => set("What is 84 * 3 / 2?") },
		{ label: "Explain something", apply: () => set("Explain quantum computing in simple terms.") },
	]);

	function set(value: string): void {
		text.value = value;
	}

	// Unlike the other panels this one never runs on its own: a generation is
	// tens of forward passes, which is not something to start on a keystroke or
	// on a tab switch.
	runners.set("diffusion", () => {
		client.preload("diffusion", settings()).catch(reportError);
	});
}

setupRouting();
setupLinting();
setupFillMask();
setupDiffusion();
runners.get(active)?.();
