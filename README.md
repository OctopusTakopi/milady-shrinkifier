# Milady Shrinkifier

*Protecting your timeline from the egregore since 2026.*

## Why

Some people find that a significant percentage of their timeline consists of accounts using aesthetically identical chibi avatars posting aesthetically identical content. This extension addresses that.

## How It Works

A bundled ONNX classifier scans avatars as you scroll. When it spots a match, you pick what happens:

- **Hide** — collapsed behind a click-to-reveal row.
- **Fade** — visible but at half opacity.
- **Debug** — borders and confidence scores on every post.
- **Off** — does nothing.

The popup tracks session stats (posts scanned, match rate, last sighting), keeps a list of detected accounts you can exempt individually, and collects avatar data you can export for offline labeling.

Everything runs locally. No server calls, no telemetry, nothing leaves your browser unless you explicitly export it.

## Install

There is no Chrome Web Store release yet.

To run it locally:

1. Install JavaScript dependencies:
   ```bash
   pnpm install
   ```
2. Install Python dependencies for the classifier and labeling pipeline:
   ```bash
   uv sync
   ```
3. Build the extension:
   ```bash
   pnpm run build
   ```
4. Load `dist/` as an unpacked extension in Chrome.

## Notes

- Development, debugging, and training workflow commands live in `DEVELOPMENT.md`.
- Runtime model artifacts live in `public/models/` and `public/generated/`.
- Training runs, labels, downloaded avatars, and dataset manifests live under ignored `cache/`.
- The review app supports both individual labeling and 9-up batch labeling.
- The extension runtime is ONNX-only.
