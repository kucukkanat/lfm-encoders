import { createReadStream, statSync } from "node:fs";
import type { IncomingMessage, ServerResponse } from "node:http";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

const MODEL_ROOT = fileURLToPath(new URL("../../models", import.meta.url));
const MIME: Record<string, string> = {
	".json": "application/json",
	".onnx": "application/octet-stream",
	".onnx_data": "application/octet-stream",
};

/**
 * Serve the exported model repos at /models without copying them.
 *
 * They live outside this app and run to several gigabytes, so Vite's `public/`
 * directory is the wrong tool — a production build would duplicate the lot.
 * Streaming them from disk keeps `vite build` producing just the app, and the
 * same URL shape works in dev and preview.
 */
function serveModels(): Plugin {
	const middleware = (
		request: IncomingMessage,
		response: ServerResponse,
		next: () => void,
	): void => {
		const url = request.url ?? "";
		if (!url.startsWith("/models/")) {
			next();
			return;
		}

		// normalize() collapses any ../ before it can escape MODEL_ROOT.
		const relative = normalize(decodeURIComponent(url.slice("/models/".length)));
		const path = join(MODEL_ROOT, relative);
		if (!path.startsWith(MODEL_ROOT)) {
			response.statusCode = 403;
			response.end("forbidden");
			return;
		}
		try {
			const stat = statSync(path);
			response.setHeader("Content-Type", MIME[extname(path)] ?? "application/octet-stream");
			response.setHeader("Content-Length", String(stat.size));
			// Weights are content-addressed by dtype filename and never mutate.
			response.setHeader("Cache-Control", "public, max-age=31536000, immutable");
			createReadStream(path).pipe(response);
		} catch {
			response.statusCode = 404;
			response.end(`not found: ${relative}\nRun \`bun run export\` from the repo root.`);
		}
	};

	return {
		name: "serve-models",
		configureServer: (server) => {
			server.middlewares.use(middleware);
		},
		configurePreviewServer: (server) => {
			server.middlewares.use(middleware);
		},
	};
}

export default defineConfig({
	plugins: [serveModels()],
	worker: { format: "es" },
	optimizeDeps: { exclude: ["@huggingface/transformers"] },
	server: {
		headers: {
			// onnxruntime-web's threaded WASM build needs SharedArrayBuffer, which
			// browsers only expose to cross-origin-isolated pages.
			"Cross-Origin-Opener-Policy": "same-origin",
			"Cross-Origin-Embedder-Policy": "require-corp",
		},
	},
});
