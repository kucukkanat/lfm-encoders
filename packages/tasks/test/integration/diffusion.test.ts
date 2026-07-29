import { afterAll, describe, expect, setDefaultTimeout, test } from "bun:test";
import { DIFFUSION_MODEL, type Diffuser, loadDiffuser } from "@lfm-encoder/tasks";
import { type DiffusionCase, fixtures, hasFp32, hasModels, MODEL_ROOT } from "./fixtures.js";

setDefaultTimeout(300_000);

/**
 * The MDLM decode loop against generations captured from the fp32 PyTorch
 * model. Nothing is mocked.
 *
 * At fp32 the claim is exact: the ONNX graph and the reference produce the same
 * token ids, so any drift in the schedule — an off-by-one in the block sweep, a
 * mis-ordered confidence sort — fails here. q8 is held to the weaker claim that
 * it still answers the question, because a quantized generative loop compounds
 * its own error and is not expected to reproduce a token stream.
 */
const available = hasModels(DIFFUSION_MODEL);
const cases = available ? fixtures<DiffusionCase>(DIFFUSION_MODEL).cases : [];

const loaded: Diffuser[] = [];
async function diffuser(dtype: "fp32" | "q8"): Promise<Diffuser> {
	const model = await loadDiffuser({ modelRoot: MODEL_ROOT, dtype });
	loaded.push(model);
	return model;
}

afterAll(async () => {
	await Promise.all(loaded.map((model) => model.dispose()));
});

describe.if(available)("masked diffusion", () => {
	test.if(hasFp32(DIFFUSION_MODEL))("fp32 reproduces the reference token stream", async () => {
		const model = await diffuser("fp32");
		for (const scenario of cases) {
			const result = await model.generate(scenario.prompt, {
				maxNewTokens: scenario.max_new,
				steps: scenario.steps,
				blockSize: scenario.block_size,
				temperature: 0,
			});
			expect(result.text).toBe(trimReference(scenario.reference_text));
		}
	});

	test("q8 still answers the question", async () => {
		const model = await diffuser("q8");
		const capital = cases.find((scenario) => /capital of France/i.test(scenario.prompt));
		expect(capital).toBeDefined();
		const result = await model.generate((capital as DiffusionCase).prompt, {
			maxNewTokens: 32,
			steps: 16,
			temperature: 0,
		});
		expect(result.text).toContain("Paris");
		expect(result.steps).toBeGreaterThan(0);
	});

	test("frames only ever add tokens", async () => {
		const model = await diffuser("q8");
		const filled: number[] = [];
		await model.generate("What is 12 times 8?", {
			maxNewTokens: 16,
			steps: 8,
			temperature: 0,
			onFrame: (frame) => filled.push(frame.tokens.filter((token) => token !== null).length),
		});
		// Committed slots are never taken back, so the count is monotonic and the
		// first frame is the empty canvas.
		expect(filled[0]).toBe(0);
		for (let i = 1; i < filled.length; i++) {
			expect(filled[i] as number).toBeGreaterThan(filled[i - 1] as number);
		}
	});

	test("an aborted signal stops the loop", async () => {
		const model = await diffuser("q8");
		const signal = { aborted: false };
		let frames = 0;
		const result = await model.generate("Explain gravity.", {
			maxNewTokens: 64,
			steps: 32,
			temperature: 0,
			signal,
			onFrame: () => {
				frames++;
				if (frames >= 2) signal.aborted = true;
			},
		});
		expect(result.steps).toBeLessThan(32);
	});
});

/** The reference decode keeps the role markers; the library trims them. */
function trimReference(text: string): string {
	return text.split("[/Answer]")[0]?.trim() ?? "";
}
