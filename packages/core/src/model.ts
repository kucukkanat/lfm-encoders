import {
	AutoTokenizer,
	env,
	PreTrainedModel,
	type PreTrainedTokenizer,
	Tensor,
} from "@huggingface/transformers";
import { tokenizeWithSpans } from "./offsets.js";
import type { Dtype, LoadOptions, Matrix, ModelConfig, Tokenized } from "./types.js";

/** A loaded encoder: tokenizer, ONNX session, and the metadata tying them together. */
export interface EncoderModel {
	readonly id: string;
	readonly dtype: Dtype;
	readonly config: ModelConfig;
	readonly tokenizer: PreTrainedTokenizer;
	/** Split `text` into ids with character spans. Cheap; no session involved. */
	tokenize(text: string): Tokenized;
	/** One bidirectional forward pass. Keys are the graph's output names. */
	forward(tokenized: Tokenized): Promise<Readonly<Record<string, Matrix>>>;
	dispose(): Promise<void>;
}

const DEFAULT_DTYPE: Dtype = "q8";

export async function loadEncoderModel(
	id: string,
	options: LoadOptions = {},
): Promise<EncoderModel> {
	const dtype = options.dtype ?? DEFAULT_DTYPE;
	applyModelRoot(options.modelRoot);

	const { onProgress } = options;
	const reporting = onProgress
		? {
				progress_callback: (event: {
					status: string;
					file?: string;
					loaded?: number;
					total?: number;
				}) => {
					if (event.status !== "progress") return;
					const total = event.total ?? 0;
					onProgress({
						file: event.file ?? "",
						loaded: event.loaded ?? 0,
						total,
						fraction: total > 0 ? (event.loaded ?? 0) / total : undefined,
					});
				},
			}
		: {};

	// Deliberately `PreTrainedModel` and not `AutoModel`: this graph is a plain
	// "feed the named inputs, read the named outputs" session, not one of
	// transformers.js's built-in architectures. Going through AutoModel would
	// reach the same place only after failing a mapping lookup and logging a
	// warning about an unknown model class.
	const [tokenizer, model] = await Promise.all([
		AutoTokenizer.from_pretrained(id, reporting),
		PreTrainedModel.from_pretrained(id, {
			dtype,
			device: options.device ?? "auto",
			...reporting,
		}),
	]);

	const config = readConfig(model.config as unknown as Record<string, unknown>);

	return {
		id,
		dtype,
		config,
		tokenizer,
		tokenize: (text) => tokenizeWithSpans(tokenizer, text),
		forward: async (tokenized) => {
			const ids = BigInt64Array.from(tokenized.ids, BigInt);
			const dims = [1, tokenized.ids.length];
			const outputs = await model({
				input_ids: new Tensor("int64", ids, dims),
				attention_mask: new Tensor("int64", new BigInt64Array(ids.length).fill(1n), dims),
			});
			return Object.fromEntries(
				config.outputs.map((name) => {
					const tensor = outputs[name];
					if (tensor === undefined) {
						throw new Error(`model '${id}' produced no output named '${name}'`);
					}
					return [name, toMatrix(tensor)];
				}),
			);
		},
		dispose: async () => {
			await model.dispose();
		},
	};
}

/**
 * Point transformers.js at a directory of exported repos.
 *
 * In the browser this is a URL prefix and the files are fetched; in Node it is
 * a filesystem path. Leaving it unset keeps the Hugging Face Hub default. This
 * writes to a module-level singleton inside transformers.js, so a process can
 * only have one root at a time — which is why it is a load option rather than
 * something callers are expected to configure themselves.
 */
function applyModelRoot(modelRoot: string | undefined): void {
	if (modelRoot === undefined) return;
	env.localModelPath = modelRoot;
	env.allowLocalModels = true;
	env.allowRemoteModels = false;
}

function readConfig(raw: Record<string, unknown>): ModelConfig {
	const onnx = (raw.onnx ?? {}) as { outputs?: string[] };
	const head = raw.head as Record<string, unknown> | undefined;
	return {
		task: String(raw.task ?? "unknown"),
		sourceModel: String(raw.source_model ?? ""),
		hiddenSize: Number(raw.hidden_size ?? 0),
		vocabSize: Number(raw.vocab_size ?? 0),
		realVocabSize: Number(raw.real_vocab_size ?? raw.vocab_size ?? 0),
		outputs: onnx.outputs ?? [],
		// Only the task heads carry this block; the plain encoder has no head.
		...(head === undefined
			? {}
			: {
					head: {
						kind: head.kind as "cosine" | "dot",
						normalize: Boolean(head.normalize),
						scale: Number(head.scale),
						bias: Number(head.bias),
						activation: head.activation as "softmax" | "sigmoid",
						prefixHeading: String(head.prefix_heading),
						projDim: Number(head.proj_dim),
					},
				}),
	};
}

/**
 * Reinterpret a `(1, tokens, features)` output as a tokens x features matrix.
 * Batch is always 1 here — every task runs a single sequence — so the leading
 * axis is dropped rather than modelled.
 */
function toMatrix(tensor: { dims: readonly number[]; data: unknown }): Matrix {
	const dims = tensor.dims;
	const cols = dims[dims.length - 1] as number;
	const rows = dims[dims.length - 2] as number;
	const data =
		tensor.data instanceof Float32Array
			? tensor.data
			: Float32Array.from(tensor.data as ArrayLike<number>);
	return {
		rows,
		cols,
		data,
		row: (index) => data.subarray(index * cols, (index + 1) * cols),
	};
}
