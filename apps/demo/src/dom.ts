/** Small typed DOM helpers, so the panels below stay about behaviour. */

export function el<K extends keyof HTMLElementTagNameMap>(
	tag: K,
	className?: string,
	text?: string,
): HTMLElementTagNameMap[K] {
	const node = document.createElement(tag);
	if (className) node.className = className;
	if (text !== undefined) node.textContent = text;
	return node;
}

export function need<T extends Element = HTMLElement>(id: string): T {
	const node = document.getElementById(id);
	if (!node) throw new Error(`missing element #${id}`);
	return node as unknown as T;
}

export function replace(parent: Element, children: readonly Node[]): void {
	parent.replaceChildren(...children);
}

/**
 * Coalesce bursts of calls into one trailing call.
 *
 * Panels re-run on every keystroke; without this, typing a sentence would queue
 * a forward pass per character and the worker would fall minutes behind.
 */
export function debounce<A extends unknown[]>(
	delayMs: number,
	fn: (...args: A) => void,
): (...args: A) => void {
	let timer: ReturnType<typeof setTimeout> | undefined;
	return (...args: A) => {
		if (timer !== undefined) clearTimeout(timer);
		timer = setTimeout(() => fn(...args), delayMs);
	};
}

export interface ChipListOptions {
	/** Give each chip a colour from the categorical token ramp. */
	readonly colored?: boolean;
	readonly onChange: (items: string[]) => void;
}

/** An editable list of free-text labels, rendered as removable chips. */
export function chipList(
	container: HTMLElement,
	form: HTMLFormElement,
	initial: readonly string[],
	options: ChipListOptions,
): { items: () => string[]; set: (items: readonly string[]) => void } {
	let items = [...initial];

	const render = (): void => {
		replace(
			container,
			items.map((label, index) => {
				const chip = el("li", "chip");
				if (options.colored) chip.style.setProperty("--chip-color", ruleColor(index));
				chip.append(el("span", undefined, label));
				const remove = el("button", "chip__remove", "×");
				remove.type = "button";
				remove.title = `Remove "${label}"`;
				remove.setAttribute("aria-label", `Remove ${label}`);
				remove.addEventListener("click", () => {
					items = items.filter((_, i) => i !== index);
					render();
					options.onChange(items);
				});
				chip.append(remove);
				return chip;
			}),
		);
	};

	form.addEventListener("submit", (event) => {
		event.preventDefault();
		const input = form.elements.namedItem("label");
		if (!(input instanceof HTMLInputElement)) return;
		const value = input.value.trim();
		if (value === "" || items.includes(value)) return;
		items = [...items, value];
		input.value = "";
		render();
		options.onChange(items);
	});

	render();
	return {
		items: () => items,
		set: (next) => {
			items = [...next];
			render();
			options.onChange(items);
		},
	};
}

/** Cycles through the six categorical tokens so any rule count stays legible. */
export function ruleColor(index: number): string {
	return `var(--rule-${(index % 6) + 1})`;
}

/** A row of one-click example buttons. */
export function examples(
	container: HTMLElement,
	entries: readonly { label: string; apply: () => void }[],
): void {
	replace(
		container,
		entries.map(({ label, apply }) => {
			const button = el("button", "example", label);
			button.type = "button";
			button.addEventListener("click", apply);
			return button;
		}),
	);
}

export function percent(value: number): string {
	return `${(value * 100).toFixed(1)}%`;
}
