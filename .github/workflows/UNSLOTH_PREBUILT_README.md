# Unsloth whisper.cpp prebuilt release CI

This repository is the Unsloth fork of [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp).
Its only job beyond mirroring upstream is to build and publish **prebuilt
`whisper-server` binaries** so Unsloth Studio can install a working local
speech-to-text engine without a C/C++ toolchain, mirroring the existing
[`unslothai/llama.cpp`](https://github.com/unslothai/llama.cpp) prebuilt pipeline.

The Studio installer (`studio/install_whisper_prebuilt.py`, default
`--published-repo unslothai/whisper.cpp`) downloads the release assets, verifies
them against in-tree pins, and drops `whisper-server` into the managed
`whisper.cpp/build/bin` directory.

## What lives here

```
.github/workflows/
  unsloth-prebuilt.yml            orchestrator: resolve tag -> fan out -> gate -> publish
  unsloth-prebuilt-cpu.yml        P0  Linux x64/arm64 + Windows x64/arm64 CPU (x86 dynamic, arm64 static)
  unsloth-prebuilt-macos.yml      P0  macOS arm64 Metal + macOS x64 CPU
  unsloth-prebuilt-cuda.yml       P1  Linux CUDA (x64 + arm64, coverage profiles)
  unsloth-prebuilt-cuda-windows.yml P1 Windows x64 CUDA (coverage profiles)
  unsloth-prebuilt-vulkan.yml     P1  Linux + Windows x64 Vulkan
  unsloth-prebuilt-rocm.yml       P1  Linux + Windows x64 ROCm (per gfx family)
scripts/
  package_bundle.py               curate one bundle + aggregate the manifest/sha256
  validate_bundle.py              CI gate: --help, transcription, --no-gpu, closure
```

## Creating the fork

1. Fork `ggml-org/whisper.cpp` into the `unslothai` org as `whisper.cpp` (or
   create the repo and push a mirror pinned to the `v1.9.1` commit).
2. Add these files at the repo root, preserving the paths above.
3. `Settings -> Actions -> General`: allow GitHub Actions, and set workflow
   permissions to **Read and write** (the orchestrator needs `contents: write`
   to create the release).
4. No extra org secrets are required for the P0 (CPU + macOS) release: the
   built-in `GITHUB_TOKEN` publishes the release and reads upstream releases.
   See "Runners and secrets you must provide" for the P1 GPU tiers.

## Triggering a release

Run the **Unsloth whisper prebuilt (full release)** workflow
(`workflow_dispatch`) with:

| input | default | meaning |
| --- | --- | --- |
| `upstream_tag` | `v1.9.1` | ggml-org/whisper.cpp tag to build |
| `packaging_suffix` | `unsloth.1` | appended to the tag -> `v1.9.1-unsloth.1` |
| `publish` | `false` | `false` builds + uploads workflow artifacts only; `true` publishes a GitHub Release |

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

## Matrix tiers

**P0 (required to publish)** - the release is blocked until all five build and
validate:

| asset | runner |
| --- | --- |
| `whisper-<tag>-linux-x64-cpu.tar.gz` | `ubuntu-22.04` |
| `whisper-<tag>-linux-arm64-cpu.tar.gz` | `ubuntu-22.04-arm` |
| `whisper-<tag>-windows-x64-cpu.zip` | `windows-2022` |
| `whisper-<tag>-macos-x64-cpu.tar.gz` | `macos-15-intel` |
| `whisper-<tag>-macos-arm64-metal.tar.gz` | `macos-14` |

x86-64 CPU bundles (Linux x64, Windows x64, macOS x64) are **dynamic**
(`-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON`): the ggml CPU backend is
built once per microarch (`ggml-cpu-sse42/sandybridge/haswell/skylakex/icelake
...`) and ggml dispatches to the best variant the host supports at runtime
(SSE4.2 .. AVX-512), matching upstream ggml-org/whisper.cpp and
unslothai/llama.cpp. `GGML_NATIVE=OFF` alone does NOT lower the ISA floor, so a
single static x86 build would bake in an AVX2/Haswell-2013 floor and SIGILL on
older CPUs. `arm64` CPU bundles are **static** single-file at the broad `armv8-a`
baseline (`-DGGML_CPU_ARM_ARCH=armv8-a`). macOS `arm64` is a static single-file
Metal build that embeds the shader library (`-DGGML_METAL_EMBED_LIBRARY=ON`,
BF16 on).

`whisper-<tag>-windows-arm64-cpu.zip` also builds in the CPU child, cross-compiled
on the x64 runner with the MSVC `amd64_arm64` toolset (mirroring
unslothai/llama.cpp's windows-cpu arm64 leg). It cannot be run/validated on the
x64 runner, so it ships **best-effort** (`continue-on-error`): a failed arm64 leg
is skipped and does not block the release. Windows arm64 hosts otherwise fall
back to the x64 bundle under emulation, or to Transformers STT.

**P1 (best-effort GPU)** - a P1 failure does NOT block the release; those hosts
fall back to a CPU bundle. GPU bundles are **dynamic** with `GGML_BACKEND_DL`
(the accelerator backend is a dlopen'd module), so every one also runs on CPU
with `--no-gpu` (Studio appends that during training):

| tier | assets |
| --- | --- |
| CUDA Linux x64 | `cuda12-legacy`, `cuda12-older`, `cuda12-newer`, `cuda12-portable`, `cuda13-older`, `cuda13-newer`, `cuda13-portable` |
| CUDA Linux arm64 | `cuda13-portable` |
| CUDA Windows x64 | same seven profiles as Linux x64 |
| Vulkan | `linux-x64-vulkan`, `windows-x64-vulkan` |
| ROCm Linux x64 | `rocm-gfx908`, `rocm-gfx90a`, `rocm-gfx103X`, `rocm-gfx110X`, `rocm-gfx1150`, `rocm-gfx1151`, `rocm-gfx120X` |
| ROCm Windows x64 | same seven gfx families |

The CUDA coverage-profile names and ROCm gfx families are **identical to
`unslothai/llama.cpp`**, so the Studio installer can select a whisper bundle
using the same profile it already resolved for the installed llama.cpp marker.

The CUDA runtime (`libcudart`/`libcublas`) is intentionally **not** bundled;
the installer pairs it with the user's PyTorch CUDA runtime via `runtime_line`.
ROCm bundles ship their runtime libs and `rocblas`/`hipblaslt` kernel dirs
co-located with `$ORIGIN` rpath.

## Asset naming contract

```
whisper-<tag>-<os>-<arch>-<accel>.<ext>
```

- `<tag>`   packaging tag, e.g. `v1.9.1-unsloth.1`
- `<os>`    `linux` | `macos` | `windows`
- `<arch>`  `x64` | `arm64`
- `<accel>` `cpu` | `metal` | `vulkan` | a CUDA profile (`cuda12-portable`, ...) | `rocm-<gfx>` (`rocm-gfx110X`, ...)
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
  "source_repo": "unslothai/whisper.cpp",
  "source_commit": "<40-hex>",
  "upstream_repo": "ggml-org/whisper.cpp",
  "upstream_tag": "v1.9.1",
  "packaging_tag": "v1.9.1-unsloth.1",
  "generated_at_utc": "2026-...Z",
  "studio_protocol": { "...": "see below" },
  "artifacts": [
    {
      "asset": "whisper-v1.9.1-unsloth.1-linux-x64-cuda12-portable.tar.gz",
      "os": "linux", "arch": "x64", "backend": "cuda",
      "accel": "cuda12-portable", "install_kind": "linux-x64-cuda",
      "runtime_line": "cuda12", "coverage_class": "portable",
      "supported_sms": ["70","75","80","86","89","90","100","103","120"],
      "min_sm": 70, "max_sm": 120,
      "gfx_target": null, "mapped_targets": null,
      "min_os": "glibc-2.35",
      "sha256": "<hex>"
    }
  ]
}
```

CPU/Vulkan/Metal entries set `runtime_line`/`coverage_class`/`supported_sms`/
`gfx_target` to `null`; ROCm entries set `gfx_target` + `mapped_targets` (the
concrete gfx list the umbrella target compiles for). `whisper-prebuilt-sha256.json`
is a flat `name -> {kind, repo, sha256, source_commit, upstream_tag}` index
covering every bundle, the two source archives and the manifest itself.

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

This gate runs on the **P0** jobs (CPU Linux x64/arm64, Windows x64, macOS
x64/arm64), whose native runners can execute the binary. The **P1 GPU** jobs
(CUDA / ROCm / Vulkan) are built, packaged and uploaded **un-run**, exactly like
unslothai/llama.cpp's GPU release jobs: their GitHub-hosted build runners have no
GPU and no NVIDIA/AMD driver, so a co-located GPU backend module cannot resolve
its host driver there and there is nothing meaningful to validate. **Real GPU
inference** (step 3 with a live device) requires a self-hosted accelerator runner
and is not wired into the default matrix - see below.

## Runners and secrets you must provide

Everything below is the repo owner's to supply; confirm before relying on it.

- **P0 needs no extra secrets or self-hosted runners.** It uses only
  GitHub-hosted runners (`ubuntu-22.04`, `ubuntu-22.04-arm`, `windows-2022`,
  `macos-14`, `macos-15-intel`) and the built-in `GITHUB_TOKEN`.
- **Runner label assumptions** (confirm they exist in your org / adjust if the
  images have been renamed): `ubuntu-22.04`, `ubuntu-22.04-arm`,
  `ubuntu-24.04-arm` (CUDA arm64), `windows-2022`, `windows-latest` (ROCm),
  `macos-14` (arm64 Metal), `macos-15-intel` (x64).
- **Real GPU validation is NOT enabled by default.** To run
  `validate_bundle.py --gpu` (a live transcription on a device) you must add
  self-hosted runners labelled for each accelerator (e.g. an NVIDIA runner, an
  AMD ROCm runner, a Vulkan-capable runner) and add a job that downloads the
  built bundle artifact and runs `validate_bundle.py --gpu` on that runner.
  Until then GPU bundles ship without a build-time runtime gate (built and
  packaged only), as with unslothai/llama.cpp.
- **Code signing / notarization is NOT configured.** macOS bundles are unsigned
  and unnotarized, and Windows binaries are unsigned. If Studio needs signed
  binaries, add your Apple Developer ID / Authenticode secrets and signing steps
  to the macOS and Windows children.
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
