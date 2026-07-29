/**
 * Masked diffusion (MDLM) text generation on the bidirectional encoder.
 *
 * A port of [LiquidAI/masked-diffusion](https://huggingface.co/spaces/LiquidAI/masked-diffusion)
 * to ONNX + transformers.js, so the loop runs in the tab instead of on a server.
 *
 * The model is not a decoder and has no KV cache to roll forward. Generation
 * instead starts from a *canvas*: the prompt followed by `maxNewTokens` copies
 * of `<|mask|>`. Every pass predicts all of the still-masked positions at once,
 * and only the most confident of them are committed; the rest go back to being
 * masks and are re-predicted next pass, now conditioned on what was just
 * written — to their right as well as their left, which is the part a causal
 * decoder cannot do.
 *
 * Two details make the output legible rather than mush:
 *
 *   * **Blocks.** Unmasking is confined to a `blockSize` window that sweeps
 *     left to right. Without it the model commits scattered high-confidence
 *     tokens across the whole canvas — punctuation and stopwords — and then has
 *     to write prose around them.
 *   * **Adjacency.** Within a pass, two neighbouring positions are never
 *     committed together. Each was predicted while the other was still a mask,
 *     so both are individually likely and jointly often not ("the the").
 *
 * The reference implementation additionally caches K/V and shortconv state so
 * later passes recompute only the active block. That needs a graph with cache
 * inputs; this one re-runs the full canvas each pass instead, which is simpler,
 * exact, and costs a constant ~`T/blockSize` factor in compute.
 */
import { type EncoderModel, type LoadOptions, loadEncoderModel } from "@lfm-encoder/core";

export const DIFFUSION_MODEL = "kucukkanat/LFM2.5-Encoder-350M-Diffusion-ONNX";

/** One denoising pass, as the UI wants to draw it. */
export interface DiffusionFrame {
	readonly step: number;
	/** One entry per generated slot; `null` while the slot is still masked. */
	readonly tokens: readonly (string | null)[];
	/** Slots committed by *this* step, for highlighting the delta. */
	readonly revealed: readonly number[];
}

export interface DiffusionOptions {
	/** Size of the canvas, in tokens. Generation cannot exceed it. */
	readonly maxNewTokens?: number;
	/** Total denoising passes, spread evenly across the blocks. */
	readonly steps?: number;
	readonly blockSize?: number;
	/** 0 is greedy; higher values sample via Gumbel noise on the logits. */
	readonly temperature?: number;
	/** Confidence at which a token is committed ahead of the step budget. */
	readonly tau?: number;
	readonly system?: string;
	/** Called after every pass, including the initial all-masked frame. */
	readonly onFrame?: (frame: DiffusionFrame) => void;
	/** Checked between passes; generation stops early and returns what it has. */
	readonly signal?: { readonly aborted: boolean };
}

export interface DiffusionResult {
	/** The answer, trimmed at the first role marker. */
	readonly text: string;
	/** Denoising passes actually run — fewer than `steps` when `tau` bites. */
	readonly steps: number;
	readonly promptTokens: number;
	readonly canvasTokens: number;
}

export interface Diffuser {
	readonly model: EncoderModel;
	generate(prompt: string, options?: DiffusionOptions): Promise<DiffusionResult>;
	dispose(): Promise<void>;
}

export async function loadDiffuser(options: LoadOptions = {}): Promise<Diffuser> {
	const model = await loadEncoderModel(DIFFUSION_MODEL, options);
	const config = model.config.diffusion;
	if (config === undefined) {
		throw new Error(`model '${DIFFUSION_MODEL}' carries no diffusion schedule`);
	}
	// Ids at or above this are vocabulary padding the checkpoint never trained;
	// they must be excluded from both the argmax and the softmax denominator, or
	// the confidences are not comparable to the reference implementation's.
	const vocab = Math.min(model.config.realVocabSize || config.maskTokenId, model.config.vocabSize);

	return {
		model,
		generate: (prompt, opts = {}) => diffuse(model, config, vocab, prompt, opts),
		dispose: () => model.dispose(),
	};
}

type Schedule = NonNullable<EncoderModel["config"]["diffusion"]>;

/** Lay the prompt out the way the diffusion-SFT run did. */
export function renderChatPrompt(template: Schedule["template"], user: string, system?: string) {
	const head = system?.trim() ? template.system.replace("{system}", system.trim()) : "";
	return head + template.user.replace("{user}", user.trim());
}

/** Cut the decoded canvas at the first role marker the model wrote. */
export function trimAnswer(text: string, stops: readonly string[]): string {
	let out = text;
	for (const stop of stops) {
		const at = out.indexOf(stop);
		if (at !== -1) out = out.slice(0, at);
	}
	return out
		.trim()
		.replace(/^(?:\[Answer\]|Answer\b)\s*:?\s*/i, "")
		.replace(/\s*(?:\.{3}|…)\s*$/, "")
		.trim();
}

async function diffuse(
	model: EncoderModel,
	schedule: Schedule,
	vocab: number,
	prompt: string,
	options: DiffusionOptions,
): Promise<DiffusionResult> {
	const maxNew = options.maxNewTokens ?? schedule.maxNewTokens;
	const totalSteps = options.steps ?? schedule.steps;
	const blockSize = options.blockSize ?? schedule.blockSize;
	const temperature = options.temperature ?? schedule.temperature;
	const tau = options.tau ?? schedule.tau;
	const mask = schedule.maskTokenId;

	const promptIds = model.tokenize(renderChatPrompt(schedule.template, prompt, options.system)).ids;
	const start = promptIds.length;
	const total = start + maxNew;
	const canvas = new Int32Array(total).fill(mask);
	canvas.set(promptIds, 0);

	// Special tokens are skipped for *display* only: the model pads the tail of
	// the canvas with `<|pad|>`, and rendering those literally would fill the
	// animation with noise the final answer does not contain.
	const decodeSlot = (id: number) =>
		id === mask ? null : model.tokenizer.decode([id], { skip_special_tokens: true });
	const view = () => Array.from(canvas.subarray(start), decodeSlot);
	options.onFrame?.({ step: 0, tokens: view(), revealed: [] });

	const blocks = Math.max(1, Math.ceil(maxNew / blockSize));
	const perBlock = Math.max(1, Math.ceil(totalSteps / blocks));
	let step = 0;

	for (let block = 0; block < blocks; block++) {
		const from = start + block * blockSize;
		const to = Math.min(start + (block + 1) * blockSize, total);
		let budget = perBlock;

		for (;;) {
			if (options.signal?.aborted) return finish();
			const rows: number[] = [];
			for (let i = from; i < to; i++) if (canvas[i] === mask) rows.push(i);
			if (rows.length === 0) break;

			const { logits } = await model.forwardIds(canvas);
			if (logits === undefined) {
				throw new Error(`model '${model.id}' produced no logits`);
			}
			const candidates = rows.map((position) => ({
				position,
				...select(logits.row(position).subarray(0, vocab), temperature),
			}));
			candidates.sort((a, b) => b.confidence - a.confidence);

			// Every pass must commit *something*, or a low-confidence block would
			// loop forever; `minimum` is what is left to write divided by the passes
			// left to write it in, so the block always lands inside its budget.
			const minimum = budget <= 1 ? rows.length : Math.ceil(rows.length / budget);
			const taken = new Set<number>();
			const revealed: number[] = [];
			const commit = (candidate: (typeof candidates)[number]) => {
				canvas[candidate.position] = candidate.id;
				taken.add(candidate.position);
				revealed.push(candidate.position - start);
			};

			for (const [index, candidate] of candidates.entries()) {
				const adjacent = taken.has(candidate.position - 1) || taken.has(candidate.position + 1);
				const eligible = candidate.confidence >= tau || revealed.length < minimum;
				// The single best candidate always goes in, adjacency or not: it is
				// what guarantees forward progress when everything is neighbouring.
				if (index === 0 || (eligible && !adjacent)) commit(candidate);
			}
			// Adjacency may have starved the quota; top it up in confidence order.
			for (const candidate of candidates) {
				if (revealed.length >= minimum) break;
				if (!taken.has(candidate.position)) commit(candidate);
			}

			budget = Math.max(1, budget - 1);
			step++;
			options.onFrame?.({ step, tokens: view(), revealed });
		}
	}
	return finish();

	function finish(): DiffusionResult {
		const written = Array.from(canvas.subarray(start)).filter((id) => id !== mask);
		const text = trimAnswer(
			model.tokenizer.decode(written, { skip_special_tokens: true }),
			schedule.template.stop,
		);
		return { text, steps: step, promptTokens: start, canvasTokens: total };
	}
}

/** Pick this position's token and report how sure the model was. */
function select(row: Float32Array, temperature: number): { id: number; confidence: number } {
	let id = 0;
	let best = Number.NEGATIVE_INFINITY;
	for (let i = 0; i < row.length; i++) {
		// Gumbel-max: adding Gumbel noise to the logits and taking the argmax is
		// exactly a draw from the tempered softmax, without building the CDF.
		const value =
			temperature > 0
				? (row[i] as number) + temperature * -Math.log(-Math.log(Math.random() || 1e-20))
				: (row[i] as number);
		if (value > best) {
			best = value;
			id = i;
		}
	}
	// Confidence is the *unperturbed* probability of whatever was chosen — the
	// scheduler ranks candidates by it, so it has to mean the same thing at every
	// temperature.
	let max = Number.NEGATIVE_INFINITY;
	for (let i = 0; i < row.length; i++) if ((row[i] as number) > max) max = row[i] as number;
	let sum = 0;
	for (let i = 0; i < row.length; i++) sum += Math.exp((row[i] as number) - max);
	return { id, confidence: Math.exp((row[id] as number) - max) / sum };
}
