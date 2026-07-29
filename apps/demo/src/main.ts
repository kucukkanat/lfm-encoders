import type { Device, Dtype } from "@lfm-encoder/core";
import type { LintWord, MaskSlot, RouteResult } from "@lfm-encoder/tasks";
import "./app.css";
import { InferenceClient, type LintData, latest } from "./client.js";
import { chipList, debounce, el, examples, need, percent, replace, ruleColor } from "./dom.js";
import type { Settings, TaskName } from "./worker.js";

const client = new InferenceClient();

// ---------------------------------------------------------------- settings

const dtypeSelect = need<HTMLSelectElement>("dtype");
const deviceSelect = need<HTMLSelectElement>("device");
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
	status(`${readyNote} — cached for next time.`);
};
client.onTiming = (elapsedMs) => {
	status(`${readyNote} · forward pass ${elapsedMs.toFixed(0)}ms`);
};

function reportError(error: unknown): void {
	meter.hidden = true;
	const message = error instanceof Error ? error.message : String(error);
	status(
		/not found|404/i.test(message)
			? `Missing weights. Run \`bun run export\` from the repo root. (${message})`
			: message,
		"error",
	);
}

// -------------------------------------------------------------------- tabs

const panels = new Map<TaskName, HTMLElement>([
	["routing", need("panel-routing")],
	["linting", need("panel-linting")],
	["fill-mask", need("panel-fill-mask")],
]);
const runners = new Map<TaskName, () => void>();
let active: TaskName = "routing";

for (const tab of need("tabs").querySelectorAll<HTMLButtonElement>(".tab")) {
	tab.addEventListener("click", () => {
		active = tab.dataset.panel as TaskName;
		for (const other of need("tabs").querySelectorAll<HTMLButtonElement>(".tab")) {
			other.setAttribute("aria-selected", String(other === tab));
		}
		for (const [name, panel] of panels) panel.hidden = name !== active;
		runners.get(active)?.();
	});
}

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

	const infer = latest((body: string, categories: string[]) =>
		client.route(settings(), body, categories),
	);

	const run = debounce(250, () => {
		const body = text.value.trim();
		const categories = labels.items();
		if (body === "" || categories.length === 0) {
			replace(output, [el("li", "empty", "Add some text and at least one category.")]);
			return;
		}
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
	runners.set("routing", run);
}

// --------------------------------------------------------------- linting

function setupLinting(): void {
	const text = need<HTMLTextAreaElement>("lint-text");
	const output = need("lint-output");
	const legend = need("lint-legend");
	const meta = need("lint-meta");
	const threshold = need<HTMLInputElement>("lint-threshold");
	const thresholdValue = need<HTMLOutputElement>("lint-threshold-value");

	const POLICY = [
		"no guarantees about financial returns",
		"no medical or health claims",
		"no disparaging competitors by name",
	];
	const rules = chipList(need("lint-rules"), need<HTMLFormElement>("lint-add"), POLICY, {
		colored: true,
		onChange: () => run(),
	});

	const infer = latest((body: string, policy: string[]) => client.lint(settings(), body, policy));
	let current: LintData | undefined;

	const run = debounce(250, () => {
		const body = text.value.trim();
		const policy = rules.items();
		if (body === "" || policy.length === 0) {
			replace(output, [el("span", "empty", "Add a draft and at least one rule.")]);
			replace(legend, []);
			return;
		}
		infer(body, policy)
			.then((result) => {
				if (!result) return;
				current = result;
				render(text.value.trim());
			})
			.catch(reportError);
	});

	/**
	 * Rebuild the highlighted text from word spans.
	 *
	 * The spans index the original string, so the gaps between them — the
	 * whitespace and punctuation the model never scored — are copied through
	 * verbatim. That keeps the output character-identical to the input.
	 */
	function render(body: string): void {
		if (!current) return;
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
		render(text.value.trim());
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

	text.addEventListener("input", run);
	runners.set("linting", run);
}

// ------------------------------------------------------------- fill-mask

function setupFillMask(): void {
	const text = need<HTMLTextAreaElement>("mask-text");
	const output = need("mask-output");
	const meta = need("mask-meta");

	const infer = latest((body: string) => client.fillMask(settings(), body, 5));

	const run = debounce(400, () => {
		const body = text.value.trim();
		if (!body.includes("<|mask|>")) {
			replace(output, [el("p", "empty", "Put <|mask|> somewhere in the text.")]);
			return;
		}
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
	runners.set("fill-mask", run);
}

setupRouting();
setupLinting();
setupFillMask();
runners.get(active)?.();
