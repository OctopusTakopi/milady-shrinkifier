import { resolve } from "node:path";

import { viteStaticCopy } from "vite-plugin-static-copy";
import { defineConfig } from "vite";

export default defineConfig({
  publicDir: "public",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "es2022",
    minify: false,
    sourcemap: true,
    rollupOptions: {
      input: {
        popup: resolve(__dirname, "popup.html"),
      },
      output: {
        entryFileNames: "[name].js",
        chunkFileNames: "chunks/[name].js",
        assetFileNames: "assets/[name][extname]",
      },
    },
  },
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: "node_modules/onnxruntime-web/dist/*.wasm",
          dest: "ort",
        },
      ],
    }),
  ],
});
