# Unsloth whisper.cpp prebuilt release CI

This repository is the Unsloth fork of [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp).
Its only job beyond mirroring upstream is to build and publish **prebuilt
`whisper-server` binaries** so Unsloth Studio can install a working local
speech-to-text engine without a C/C++ toolchain, mirroring the existing
[`unslothai/llama.cpp`](https://github.com/unslothai/llama.cpp) prebuilt pipeline.

Releases are **slim-only**: every published bundle contains just
`whisper-server` + the whisper shared library + metadata, and rides the ggml of
a **paired `unslothai/llama.cpp` release**. The llama prebuilts are first-class
and always installed, and they already ship every ggml backend (CPU variants,
CUDA, HIP, Vulkan, Metal), so one slim whisper bundle per os/arch serves the
full [Windows, Linux (and WSL via Linux), macOS] x [NVIDIA, AMD, CPU-only]
product. Fat whisper bundles (per-accelerator `cpu`/`cuda*`/`rocm-*`/`vulkan`/
`metal` assets) are **no longer published**; releases that already carried them
keep theirs.

The Studio installer (`studio/install_whisper_prebuilt.py`, default
`--published-repo unslothai/whisper.cpp`) downloads the release assets, verifies
them against in-tree pins, and drops `whisper-server` into the managed
`whisper.cpp/build/bin` directory.

## What lives here

```
.github/workflows/
  unsloth-prebuilt.yml            orchestrator: resolve tag -> slim child -> gate -> publish
  unsloth-prebuilt-slim.yml       Linux + Windows + macOS, x64/arm64 slim (paired llama.cpp ggml)
scripts/
  package_bundle.py               curate one bundle + aggregate the manifest/sha256
  validate_bundle.py              CI gate: --help, transcription, --no-gpu, closure
```

The fat per-accelerator children (`unsloth-prebuilt-cpu.yml`,
`unsloth-prebuilt-macos.yml`, `unsloth-prebuilt-cuda.yml`,
`unsloth-prebuilt-cuda-windows.yml`, `unsloth-prebuilt-vulkan.yml`,
`unsloth-prebuilt-rocm.yml`) remain in the tree for reference, but the
orchestrator no longer invokes them and their bundles are no longer published.

## Creating the fork

1. Fork `ggml-org/whisper.cpp` into the `unslothai` org as `whisper.cpp` (or
   create the repo and push a mirror pinned to the `v1.9.1` commit).
2. Add these files at the repo root, preserving the paths above.
3. `Settings -> Actions -> General`: allow GitHub Actions, and set workflow
   permissions to **Read and write** (the orchestrator needs `contents: write`
   to create the release).
4. No extra org secrets are required: the built-in `GITHUB_TOKEN` publishes the
   release and reads the upstream whisper.cpp and paired llama.cpp releases.
   See "Runners and secrets you must provide" for optional extras.

## Triggering a release

Run the **Unsloth whisper prebuilt (full release)** workflow
(`workflow_dispatch`) with:

| input | default | meaning |
| --- | --- | --- |
| `upstream_tag` | `v1.9.1` | ggml-org/whisper.cpp tag to build |
| `packaging_suffix` | `unsloth.1` | appended to the tag -> `v1.9.1-unsloth.1` |
| `publish` | `false` | `false` builds + uploads workflow artifacts only; `true` publishes a GitHub Release |
| `llama_tag` | (blank) | paired `unslothai/llama.cpp` release tag for slim bundles; blank (and every scheduled run) resolves to the newest published llama release |

The `resolve` job dereferences `upstream_tag` to an **immutable commit SHA** and
records it, then stamps `cmake/build-info.cmake` (build number and commit) and
uploads the source tree once.
Every build child extracts that one tree, so a tag re-point upstream can never
change what a packaging tag shipped. Bump `packaging_suffix` to
`unsloth.2`, `unsloth.3`, ... for rebuilds of the same upstream tag.

Publication is **atomic and tiered**: the release is created as a draft (hidden
from the anonymous API the installer reads), every asset is uploaded, and only
then is it flipped to `draft=false`. A leftover draft from a failed run is
detected as "not published" and rebuilt on the next run.

## The slim bundle set

One slim bundle per os/arch, six in total:

| asset | runner | required to publish |
| --- | --- | --- |
| `whisper-<tag>-linux-x64-slim.tar.gz` | `ubuntu-22.04` | yes |
| `whisper-<tag>-linux-arm64-slim.tar.gz` | `ubuntu-22.04-arm` | yes |
| `whisper-<tag>-windows-x64-slim.zip` | `windows-2022` | yes |
| `whisper-<tag>-windows-arm64-slim.zip` | `windows-2022` (cross, `amd64_arm64`) | best-effort |
| `whisper-<tag>-macos-x64-slim.tar.gz` | `macos-15-intel` | yes |
| `whisper-<tag>-macos-arm64-slim.tar.gz` | `macos-26` | yes |

**Slim bundles** contain only `whisper-server` + the whisper shared library
(NO `libggml*`): whisper is compiled against the ggml source tree of the paired
`unslothai/llama.cpp` release (the `llama_tag` resolved above; the vendored
`ggml/` dir is swapped before configuring), and at install time the Studio
installer links every `libggml*` from that llama install's bin dir into the
whisper bin dir, where the dynamic loader (and, for dlopen modules, ggml's
registry) finds them. Because the llama install carries every ggml backend,
each slim bundle serves **every accelerator on its platform**: CUDA, HIP,
Vulkan and the CPU variants on Linux/Windows, Metal and CPU on macOS.

All slices build shared (`BUILD_SHARED_LIBS=ON`, `GGML_BACKEND_DL=ON`,
`GGML_CPU_ALL_VARIANTS=OFF`, no accelerator toolkits; the in-tree CPU backend
exists only to link the build and is discarded at packaging). The rpath is
`$ORIGIN` on Linux and `@loader_path` on macOS (Windows resolves DLLs from the
executable's directory), so the linked-in llama libraries resolve from the
bundle dir itself. macOS builds pin `-DGGML_METAL=OFF -DWHISPER_COREML=OFF`
plus the same deployment targets as the previous fat macOS jobs (arm64 `14.0`,
x64 `13.3`).

`whisper-<tag>-windows-arm64-slim.zip` is cross-compiled on the x64 runner with
clang (`cmake/arm64-windows-llvm.cmake`). It cannot be run/validated on the x64
runner, so it ships **best-effort** (`continue-on-error`): a failed arm64 leg
is skipped and does not block the release.

Manifest entries carry `install_kind: "slim"` plus `requires_llama_tag` /
`requires_ggml_commit` / `requires_ggml_version` / `requires_ggml_sonames` so
the installer can verify the pairing before wiring the bundle. Fork tags are
`b<upstream_build>-mix-<ggml_commit>`; `requires_ggml_commit` records the
`-mix-` ggml commit, which is the ABI the slim bundle links against. The
installer pairs on that ggml commit, not the whole tag, so a newer llama build
that keeps the same ggml still backs the bundle. `requires_ggml_sonames` is the exact list
of library filenames that must exist in the paired llama bin dir for the loader
to bring the bundle up in a lib-in-same-dir layout (on macOS this includes the
backend dylibs that llama's `libggml.0.dylib` itself loads via `@rpath`).

CI validates each slim bundle on its free runner with the exact consumer
wiring (`validate_bundle.py --ggml-dir`): Linux/Windows x64 against the
same-tag llama `app-<llama_tag>-<os>-<arch>-cpu` bundle, macOS against the
same-tag `llama-<llama_tag>-bin-macos-<arch>` bundle. GPU-side validation on
other accelerators is skipped, as it was for fat GPU bundles.

## Asset naming contract

```
whisper-<tag>-<os>-<arch>-<accel>.<ext>
```

- `<tag>`   packaging tag, e.g. `v1.9.1-unsloth.1`
- `<os>`    `linux` | `macos` | `windows`
- `<arch>`  `x64` | `arm64`
- `<accel>` `slim` (the only published accel since the slim-only model; old
  releases keep their fat `cpu`/`metal`/`vulkan`/`cuda*`/`rocm-*` assets)
- `<ext>`   `tar.gz` on Unix, `zip` on Windows

Each release also carries:

- `whisper-prebuilt-manifest.json`
- `whisper-prebuilt-sha256.json`
- `whisper.cpp-source-<tag>.tar.gz` and `whisper.cpp-source-commit-<sha>.tar.gz`
  (the stamped source, so a source build reproduces the same binary)

### Manifest schema (`whisper-prebuilt-manifest.json`)

```json
{
  "schema_version": 1,
  "component": "whisper.cpp",
  "paired_llama_tag": "<llama_tag>",
  "paired_ggml_commit": "<ggml_commit>",
  "source_repo": "unslothai/whisper.cpp",
  "source_commit": "<40-hex>",
  "upstream_repo": "ggml-org/whisper.cpp",
  "upstream_tag": "v1.9.1",
  "packaging_tag": "v1.9.1-unsloth.1",
  "generated_at_utc": "2026-...Z",
  "studio_protocol": { "...": "see below" },
  "artifacts": [
    {
      "asset": "whisper-v1.9.1-unsloth.1-linux-x64-slim.tar.gz",
      "os": "linux", "arch": "x64", "backend": "slim",
      "accel": "slim", "install_kind": "slim",
      "runtime_line": null, "coverage_class": null,
      "supported_sms": null, "min_sm": null, "max_sm": null,
      "gfx_target": null, "mapped_targets": null,
      "min_os": "glibc-2.35",
      "sha256": "<hex>",
      "requires_llama_tag": "<llama_tag>",
      "requires_ggml_commit": "<ggml_commit>",
      "requires_ggml_version": "0.17.0",
      "requires_ggml_sonames": ["libggml.so.0", "libggml-base.so.0"]
    }
  ]
}
```

Every published entry is a slim entry: exactly one per os/arch with
`backend: "slim"` (the consumer maps any accel, including cpu and metal, onto
it; there are no per-accel slim entries). `requires_ggml_sonames` is
per-platform: `["libggml.so.0", "libggml-base.so.0"]` on Linux, `["ggml.dll",
"ggml-base.dll"]` on Windows, and on macOS the full dylib closure the loader
needs (`libggml.0.dylib`, `libggml-base.0.dylib`, `libggml-cpu.0.dylib`,
`libggml-blas.0.dylib`, `libggml-rpc.0.dylib`, plus `libggml-metal.0.dylib` on
arm64). `min_os` follows the platform convention: `glibc-2.35`, `windows-10`,
`macos-14.0` (arm64) / `macos-13.3` (x64). The manifest's top level records the
pairing as `paired_llama_tag` and its ggml commit as `paired_ggml_commit`.
`whisper-prebuilt-sha256.json`
is a flat `name -> {kind, repo, sha256, source_commit, upstream_tag}` index
covering the six slim bundles, the two source archives and the manifest itself.

Both sidecars are generated by `scripts/package_bundle.py --emit-manifest
--emit-sha256` from the bundles' **embedded** `UNSLOTH_WHISPER_PREBUILT_INFO.json`,
so the manifest can never disagree with what was actually compiled.

### `studio_protocol`

Pins the `whisper-server` `/inference` multipart contract Studio's STT sidecar
speaks, so the installer can reject a server whose HTTP contract has drifted:

```json
{
  "version": 1,
  "endpoint": "/inference",
  "method": "POST",
  "content_type": "multipart/form-data",
  "request_fields": {
    "file": "audio file bytes (required)",
    "temperature": "float",
    "response_format": "one of json|text|srt|vtt|verbose_json",
    "beam_size": "int",
    "language": "ISO language code or 'auto'"
  },
  "response_json_field": "text"
}
```

Bump `version` (in `scripts/package_bundle.py`, `STUDIO_PROTOCOL`) on any
breaking change to what Studio sends or expects back.

### Accepted audio formats (`/inference`)

Bundles decode WAV, MP3, FLAC and Ogg Vorbis (miniaudio + stb_vorbis). Ogg/WebM
Opus (browser `MediaRecorder`, WhatsApp/Telegram voice notes) is not decodable
and returns HTTP 400; transcode first:
`ffmpeg -i in.opus -ar 16000 -ac 1 out.wav`. Studio is unaffected: it decodes
with PyAV and posts WAV.

## CI validation gate

Before a bundle is uploaded, `scripts/validate_bundle.py` runs against the
freshly built archive on its build runner:

1. `whisper-server --help` returns usage text (the binary loaded its libs).
2. Dependency-closure check (`ldd` on Linux, dyld load on macOS; skipped on
   Windows): no unresolved libraries.
3. Start `whisper-server` with a tiny English model
   (`ggerganov/whisper.cpp/ggml-tiny.en.bin`) and POST one real multipart
   `/inference` request for the repo's own `samples/jfk.wav`; assert HTTP 200
   and a non-empty JSON `text`.
4. Repeat step 3 with `--no-gpu` so the CPU fallback path is proven.

Slim bundles ship no ggml, so the gate first wires them to the paired llama
bundle (`--ggml-dir`, the exact linking the Studio installer performs) and then
runs the same checks: Linux x64/arm64 and Windows x64 against the same-tag
llama cpu bundle, macOS x64/arm64 against the same-tag llama macos bundle
(whose ggml carries the statically-registered cpu/blas/metal backends).
Windows arm64 is cross-compiled and cannot execute on its x64 build runner, so
it ships un-run and best-effort. **Real GPU inference** (step 3 with a live
device) requires a self-hosted accelerator runner and is not wired into the
default matrix - see below.

## Runners and secrets you must provide

Everything below is the repo owner's to supply; confirm before relying on it.

- **The slim release needs no extra secrets or self-hosted runners.** It uses
  only GitHub-hosted runners (`ubuntu-22.04`, `ubuntu-22.04-arm`,
  `windows-2022`, `macos-26`, `macos-15-intel`) and the built-in
  `GITHUB_TOKEN`.
- **Runner label assumptions** (confirm they exist in your org / adjust if the
  images have been renamed): `ubuntu-22.04`, `ubuntu-22.04-arm`,
  `windows-2022`, `macos-26` (arm64), `macos-15-intel` (x64).
- **Real GPU validation is NOT enabled by default.** To run
  `validate_bundle.py --gpu` (a live transcription on a device) you must add
  self-hosted runners labelled for each accelerator (e.g. an NVIDIA runner, an
  AMD ROCm runner, a Vulkan-capable runner) and add a job that downloads the
  built slim bundle artifact, wires it to the matching llama bundle with
  `--ggml-dir` and runs `validate_bundle.py --gpu` on that runner. Until then
  accelerator paths ship gated only by the CPU-side wiring proof, as with
  unslothai/llama.cpp.
- **Code signing / notarization is NOT configured.** macOS bundles are unsigned
  and unnotarized, and Windows binaries are unsigned. If Studio needs signed
  binaries, add your Apple Developer ID / Authenticode secrets and signing steps
  to the slim child's macOS and Windows jobs.
- **First release should be pinned, not followed blindly.** Publish
  `v1.9.1-unsloth.1` with `publish=true`, review the assets, then record the
  release id + per-asset sha256 in Studio's `studio/whisper_prebuilt_pins.json`
  before the installer trusts it.

## Trust and reproducibility

- Every bundle must carry the `Compiled by the Unsloth team` fingerprint
  (carried in each bundle's `BUILD_INFO.txt`, written by `package_bundle.py`);
  the orchestrator refuses to publish an unbranded bundle.
- The stamped source tarballs are shipped as release assets so Studio's
  source-build fallback reproduces the same binary (the `Compiled by the Unsloth
  team` mark is a packaging-time bundle file, not part of the binary).
- Consider enabling GitHub artifact attestations for supply-chain provenance;
  the same-origin sha256 index proves integrity, not authenticity, so Studio's
  in-tree pins remain the trust anchor.
