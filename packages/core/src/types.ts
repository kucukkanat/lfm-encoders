/** Quantization variants produced by `tools/export`. */
export type Dtype = "fp32" | "fp16" | "q8" | "q4" | "q4f16";

/**
 * Execution backend. `auto` picks WebGPU when the browser exposes it and falls
 * back to the WASM build otherwise, which is also what Node gets.
 */
export type Device = "auto" | "wasm" | "webgpu";

/** Half-open character range, in JavaScript string indices (UTF-16 code units). */
export interface Span {
	readonly start: number;
	readonly end: number;
}

/** A tokenized string, with every token anchored back to the text it came from. */
export interface Tokenized {
	readonly text: string;
	readonly ids: readonly number[];
	/** One span per id. Tokens the tokenizer inserted (BOS) are zero-width. */
	readonly spans: readonly Span[];
}

/** Row-major numeric matrix over a flat buffer, as ONNX hands them back. */
export interface Matrix {
	readonly rows: number;
	readonly cols: number;
	readonly data: Float32Array;
	/** Zero-copy view of one row. */
	row(index: number): Float32Array;
}

/** The scoring rule a two-tower task head applies to the projections. */
export interface HeadConfig {
	/** `cosine` L2-normalises both towers before scoring; `dot` does not. */
	readonly kind: "cosine" | "dot";
	readonly normalize: boolean;
	readonly scale: number;
	readonly bias: number;
	readonly activation: "softmax" | "sigmoid";
	/** Heading the model was trained to see above the label list. */
	readonly prefixHeading: string;
	readonly projDim: number;
}

/** The MDLM decode schedule, shipped in the diffusion checkpoint's config. */
export interface DiffusionConfig {
	readonly maskTokenId: number;
	readonly maskToken: string;
	readonly maxNewTokens: number;
	readonly steps: number;
	readonly blockSize: number;
	readonly temperature: number;
	/** Confidence above which a token is committed ahead of its step budget. */
	readonly tau: number;
	readonly template: {
		readonly system: string;
		readonly user: string;
		readonly stop: readonly string[];
	};
}

/** The subset of the exported `config.json` this library relies on. */
export interface ModelConfig {
	readonly task: string;
	readonly sourceModel: string;
	readonly hiddenSize: number;
	/** Padded vocabulary the logits head emits. */
	readonly vocabSize: number;
	/** Ids below this were actually trained; the rest are alignment padding. */
	readonly realVocabSize: number;
	readonly outputs: readonly string[];
	readonly head?: HeadConfig;
	readonly diffusion?: DiffusionConfig;
}

export interface LoadProgress {
	readonly file: string;
	readonly loaded: number;
	readonly total: number;
	/** 0..1, or `undefined` before the server reports a content length. */
	readonly fraction: number | undefined;
}

export interface LoadOptions {
	/**
	 * Where the exported model repos live. In the browser this is a URL prefix
	 * (`/models`); in Node it is a filesystem path. Omit to fetch from the
	 * Hugging Face Hub.
	 */
	readonly modelRoot?: string;
	/** Defaults to `q8` — see the accuracy table in the README. */
	readonly dtype?: Dtype;
	readonly device?: Device;
	readonly onProgress?: (progress: LoadProgress) => void;
}
