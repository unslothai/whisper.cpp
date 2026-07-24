#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Cross-platform packager for Unsloth whisper.cpp prebuilt bundles.

Two modes, one script (whisper's release CI has no separate assemble step):

  package   (default)  Curate a single whisper-server bundle: the executable,
                       the local dynamic-library closure it needs, the ggml
                       backend modules, ROCm kernel data dirs, licenses and an
                       embedded UNSLOTH_WHISPER_PREBUILT_INFO.json; archive to
                       whisper-<tag>-<os>-<arch>-<accel>.{tar.gz,zip}. Also emits
                       the manifest entry for the asset (stdout + <asset>.entry.json).

                       SLIM sub-mode (SLIM=1 + LLAMA_TAG + GGML_VERSION, or the
                       matching CLI flags): the bundle carries ONLY
                       whisper-server + the whisper shared library + metadata.
                       Every ggml object (base + backends) is provided at
                       runtime by the PAIRED unslothai/llama.cpp install, whose
                       ggml source tree this build compiled against; the
                       installer links those libs into the bundle dir. Curation
                       hard-fails if any libggml* would be included (mixing
                       ggml builds across snapshots is broken at the symbol
                       level), and the metadata/manifest entry gains
                       install_kind=slim + requires_llama_tag /
                       requires_ggml_version / requires_ggml_sonames.

  --emit-manifest /    Aggregate every whisper-*.{tar.gz,zip} already dropped in
  --emit-sha256        --dist into whisper-prebuilt-manifest.json (schema_version
                       1, component whisper.cpp, studio_protocol, artifacts[]) and
                       whisper-prebuilt-sha256.json. Reads each bundle's embedded
                       info, so the manifest can never disagree with what shipped.

The curation engine is OS-generic: a new OS means one PlatformStrategy (its
dependency-walk tool, lib-name convention, backend glob, archive format).

The CUDA runtime (libcudart / libcublas) is intentionally NOT bundled: the
Studio installer pairs it with the user's PyTorch runtime, selected by
runtime_line. ROCm libs, in contrast, are copied into build/bin by the ROCm
workflow and are shipped from there (co-located, $ORIGIN rpath).

Static CPU bundles have no local library closure; curation then ships the single
self-contained whisper-server binary, which is the intended single-file drop-in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# The /inference multipart contract the Studio STT sidecar speaks. Pinned into
# every manifest so the installer can reject a whisper-server whose HTTP contract
# has drifted from what Studio sends. Bump `version` on any breaking change.
STUDIO_PROTOCOL = {
    "version": 1,
    "endpoint": "/inference",
    "method": "POST",
    "content_type": "multipart/form-data",
    "request_fields": {
        "file": "audio file bytes (required)",
        "temperature": "float",
        "response_format": "one of json|text|srt|vtt|verbose_json",
        "beam_size": "int",
        "language": "ISO language code or 'auto'",
    },
    "response_json_field": "text",
}

COMPONENT = "whisper.cpp"
UPSTREAM_REPO = "ggml-org/whisper.cpp"
INFO_NAME = "UNSLOTH_WHISPER_PREBUILT_INFO.json"

# Bundles only; never the source tarballs or the sidecar JSONs.
BUNDLE_RE = re.compile(
    r"^whisper-(?P<tag>.+)-(?P<os>linux|macos|windows)-(?P<arch>x64|arm64)-"
    r"(?P<accel>[A-Za-z0-9._-]+)\.(?P<ext>tar\.gz|zip)$"
)

# Force C locale so readelf/otool output is not localized.
_C_ENV = {**os.environ, "LC_ALL": "C", "LANG": "C"}

# Fixed epoch for archive member mtimes so repeated packaging of the same tree is
# byte-stable (deterministic ordering is enforced separately by sorting names).
_FIXED_MTIME = 1704067200  # 2024-01-01T00:00:00Z


def _run(cmd: list[str]) -> str:
    # Fail loudly: a missing/erroring readelf|otool would otherwise yield an
    # empty closure and silently ship a bundle with missing libraries.
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=_C_ENV)
    except FileNotFoundError:
        sys.exit(f"ERROR: required tool '{cmd[0]}' not found")
    if r.returncode != 0:
        sys.exit(f"ERROR: {' '.join(cmd)} failed (rc={r.returncode}): {r.stderr.strip()}")
    return r.stdout


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Platform strategies
# --------------------------------------------------------------------------- #
class PlatformStrategy:
    name = "generic"
    exe_suffix = ""
    archive_ext = ".tar.gz"
    rpath = ""
    lib_suffix_re = r"\.so(\.\d+)*$"
    # Directories of runtime data (ROCm kernel libraries) shipped wholesale.
    data_dirs = ("rocblas", "hipblaslt")
    # Slim bundles ship only these whisper pieces; the paired llama.cpp install
    # provides every ggml object at runtime under these sonames. The per-build
    # GGML_SONAMES env / --ggml-sonames flag overrides the default list (used
    # on macOS, where the contract differs per arch: x64 has no metal dylib).
    whisper_lib_patterns = ("libwhisper.so*",)
    ggml_sonames = ("libggml.so.0", "libggml-base.so.0")

    def server_name(self) -> str:
        return "whisper-server" + self.exe_suffix

    def is_executable(self, path: Path) -> bool:
        if not path.is_file():
            return False
        if self.exe_suffix:
            return path.suffix.lower() == self.exe_suffix
        return not re.search(self.lib_suffix_re, path.name) and os.access(path, os.X_OK)

    def local_needed(self, path: Path, bin_dir: Path) -> list[str]:
        raise NotImplementedError

    def backend_patterns(self) -> list[str]:
        raise NotImplementedError

    def supports_symlinks(self) -> bool:
        return True

    def archive(self, stage: Path, out_path: Path) -> None:
        raise NotImplementedError


class LinuxStrategy(PlatformStrategy):
    name = "linux"
    rpath = "$ORIGIN"

    def local_needed(self, path: Path, bin_dir: Path) -> list[str]:
        needed = re.findall(r"\(NEEDED\)[^\[]*\[([^\]]+)\]", _run(["readelf", "-d", str(path)]))
        return [n for n in needed if (bin_dir / n).exists() or (bin_dir / n).is_symlink()]

    def backend_patterns(self) -> list[str]:
        return ["libwhisper.so*", "libggml*.so*",
                # ROCm runtime libs the ROCm workflow copies into build/bin.
                "libamdhip64.so*", "librocblas.so*", "libhipblas*.so*",
                "librocsolver.so*", "libamd_comgr*.so*", "libhsa-runtime64.so*",
                "librocm_sysdeps_*.so*", "librocprofiler-register.so*",
                "libroctx64.so*", "librocroller.so*", "librocm_kpack.so*",
                "libLLVM.so*", "libclang-cpp.so*", "libatomic.so*"]

    def archive(self, stage: Path, out_path: Path) -> None:
        _tar_deterministic(stage, out_path)


class MacOSStrategy(PlatformStrategy):
    name = "macos"
    rpath = "@loader_path"
    lib_suffix_re = r"\.dylib$"
    whisper_lib_patterns = ("libwhisper*.dylib",)
    # Direct references only (what whisper-server/libwhisper link). The slim
    # macOS jobs pass GGML_SONAMES with the full per-arch closure instead: the
    # paired llama macos bundles carry statically-registered backend dylibs
    # (cpu/blas/metal/rpc) that llama's libggml.0.dylib itself loads via
    # @rpath, so dyld needs those names in the bin dir too.
    ggml_sonames = ("libggml.0.dylib", "libggml-base.0.dylib")

    def local_needed(self, path: Path, bin_dir: Path) -> list[str]:
        out = _run(["otool", "-L", str(path)])
        deps: list[str] = []
        for line in out.splitlines()[1:]:  # first line echoes the file path
            m = re.match(r"\s+(\S+)\s+\(", line)
            if not m:
                continue
            ref = m.group(1)
            base = os.path.basename(ref)
            if (ref.startswith("@") or not ref.startswith("/")) and (bin_dir / base).exists():
                deps.append(base)
        return deps

    def backend_patterns(self) -> list[str]:
        # ggml builds the shared LIBRARIES (libwhisper/libggml/libggml-base) as
        # .dylib, but the dlopen'd backend MODULES (libggml-cpu-<variant>, metal,
        # blas ...) as .so on macOS: they are CMake MODULE libraries, which keep
        # the .so suffix, and ggml's loader searches for ".so" on non-Windows
        # (ggml/src/ggml-backend-reg.cpp backend_filename_extension). Ship both
        # or a dynamic bundle registers 0 backends and aborts at whisper_init.
        return ["libwhisper*.dylib", "libggml*.dylib", "libggml*.so"]

    def archive(self, stage: Path, out_path: Path) -> None:
        _tar_deterministic(stage, out_path)


class WindowsStrategy(PlatformStrategy):
    name = "windows"
    exe_suffix = ".exe"
    archive_ext = ".zip"
    rpath = ""  # Windows resolves DLLs from the executable's directory
    whisper_lib_patterns = ("whisper.dll",)
    ggml_sonames = ("ggml.dll", "ggml-base.dll")
    LOCAL_DLL_PREFIXES = ("ggml", "whisper", "amdhip", "rocblas", "hipblas",
                          "rocsolver", "amd_comgr", "rocm", "hsa")

    def local_needed(self, path: Path, bin_dir: Path) -> list[str]:
        # No portable readelf/otool on Windows runners; the project + backend
        # DLLs live beside the .exe, so ship every DLL by name convention.
        return [
            p.name for p in bin_dir.glob("*.dll")
            if p.name.lower().startswith(self.LOCAL_DLL_PREFIXES)
        ]

    def backend_patterns(self) -> list[str]:
        return ["whisper.dll", "ggml*.dll"]

    def supports_symlinks(self) -> bool:
        return False

    def archive(self, stage: Path, out_path: Path) -> None:
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(stage.rglob("*"), key=lambda x: x.relative_to(stage).as_posix()):
                if p.is_file():
                    zi = zipfile.ZipInfo(p.relative_to(stage).as_posix(),
                                         date_time=(2024, 1, 1, 0, 0, 0))
                    zi.compress_type = zipfile.ZIP_DEFLATED
                    z.writestr(zi, p.read_bytes())


def _tar_deterministic(stage: Path, out_path: Path) -> None:
    """tar.gz with sorted members and normalized metadata (reproducible)."""
    members = sorted(stage.rglob("*"), key=lambda p: p.relative_to(stage).as_posix())

    def _reset(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        ti.mtime = _FIXED_MTIME
        return ti

    with tarfile.open(out_path, "w:gz", format=tarfile.GNU_FORMAT) as tar:
        for p in members:
            arc = p.relative_to(stage).as_posix()
            tar.add(p, arcname=arc, recursive=False, filter=_reset)


STRATEGIES = {s.name: s for s in (LinuxStrategy(), MacOSStrategy(), WindowsStrategy())}


# --------------------------------------------------------------------------- #
# Curation
# --------------------------------------------------------------------------- #
def _copy_one(strategy: PlatformStrategy, bin_dir: Path, stage: Path, name: str) -> None:
    src, dst = bin_dir / name, stage / name
    if dst.exists() or dst.is_symlink():
        return
    if strategy.supports_symlinks() and src.is_symlink():
        target = os.readlink(src)
        os.symlink(target, dst)
        _copy_one(strategy, bin_dir, stage, os.path.basename(target))
    elif src.exists():
        shutil.copy2(src, dst, follow_symlinks=True)


def curate(strategy: PlatformStrategy, bin_dir: Path, stage: Path) -> None:
    server = strategy.server_name()
    if not (bin_dir / server).exists():
        sys.exit(f"ERROR: missing {bin_dir / server}")
    shutil.copy2(bin_dir / server, stage / server)
    roots: list[Path] = [stage / server]

    # ggml backend modules + whisper/ggml shared libs. Directly-linked deps also
    # surface in the NEEDED walk below; the glob is a safety net (and the only
    # way to catch dlopen'd modules).
    for pat in strategy.backend_patterns():
        for match in sorted(bin_dir.glob(pat)):
            _copy_one(strategy, bin_dir, stage, match.name)
            roots.append(stage / match.name)

    # Walk the local NEEDED closure from every root, scanning each lib once.
    queue = list(roots)
    while queue:
        for need in strategy.local_needed(queue.pop(), bin_dir):
            if not (stage / need).exists() and not (stage / need).is_symlink():
                _copy_one(strategy, bin_dir, stage, need)
                queue.append(stage / need)

    # ROCm kernel data directories (rocblas/library, hipblaslt/library): whole
    # subtrees, copied verbatim if the build placed them in bin_dir.
    for d in strategy.data_dirs:
        src = bin_dir / d
        if src.is_dir():
            shutil.copytree(src, stage / d, symlinks=strategy.supports_symlinks())


_GGML_NAME_RE = re.compile(r"^(lib)?ggml", re.I)


def curate_slim(strategy: PlatformStrategy, bin_dir: Path, stage: Path) -> None:
    """SLIM curation: whisper-server + the whisper shared library, nothing else.

    All ggml objects (base + every backend) come from the paired
    unslothai/llama.cpp install at runtime -- the installer links them into
    this bundle's directory, where ggml's registry scans for backend modules.
    Mixing ggml builds across snapshots is broken at the symbol level, so this
    is the inverse of the fat completeness check: hard-fail if any libggml*
    would ship.
    """
    server = strategy.server_name()
    if not (bin_dir / server).exists():
        sys.exit(f"ERROR: missing {bin_dir / server}")
    shutil.copy2(bin_dir / server, stage / server)

    libs: list[str] = []
    for pat in strategy.whisper_lib_patterns:
        for match in sorted(bin_dir.glob(pat)):
            _copy_one(strategy, bin_dir, stage, match.name)
            libs.append(match.name)
    if not libs:
        sys.exit(f"ERROR: slim bundle found no whisper library matching "
                 f"{strategy.whisper_lib_patterns} in {bin_dir} (slim builds must be shared)")

    offenders = sorted(p.name for p in stage.iterdir() if _GGML_NAME_RE.match(p.name))
    if offenders:
        sys.exit("ERROR: slim bundle must contain NO ggml objects, found: " + ", ".join(offenders))


def _coverage(cfg: dict) -> dict:
    """sm/gfx coverage fields for the manifest entry, by backend."""
    backend = cfg["backend"]
    if backend == "cuda":
        sms = [s for s in re.split(r"[ ;,]+", cfg.get("sms", "")) if s]
        # sm_103 (Blackwell Ultra) JIT-runs on bundled compute_100 PTX.
        if "100" in sms and "103" not in sms:
            sms = sorted([*sms, "103"], key=int)
        ints = [int(s) for s in sms] or [0]
        return {"runtime_line": cfg.get("runtime_line") or None,
                "coverage_class": cfg.get("coverage_class") or None,
                "supported_sms": sms, "min_sm": min(ints), "max_sm": max(ints),
                "gfx_target": None, "mapped_targets": None}
    if backend == "rocm":
        gfx = cfg.get("gfx_target") or cfg["accel"].removeprefix("rocm-")
        mapped = [g for g in re.split(r"[ ;,]+", cfg.get("mapped_targets", "")) if g] or [gfx]
        return {"runtime_line": "rocm", "coverage_class": None,
                "supported_sms": None, "min_sm": None, "max_sm": None,
                "gfx_target": gfx, "mapped_targets": mapped}
    # cpu / vulkan / metal: no CUDA/ROCm runtime to match.
    return {"runtime_line": None, "coverage_class": None,
            "supported_sms": None, "min_sm": None, "max_sm": None,
            "gfx_target": None, "mapped_targets": None}


def paired_ggml_commit(llama_tag: str | None) -> str | None:
    """The ggml commit a llama.cpp fork tag was built against. Fork tags are
    "b<upstream_build>-mix-<ggml_commit>"; the commit after "-mix-" fixes the
    ggml ABI the slim bundle links against, so the installer pairs on it rather
    than the build number. None when the tag has no "-mix-" marker."""
    if not llama_tag:
        return None
    marker = "-mix-"
    idx = llama_tag.rfind(marker)
    end = idx + len(marker)
    return llama_tag[end:] if idx >= 0 and end < len(llama_tag) else None


def write_metadata(stage: Path, strategy: PlatformStrategy, cfg: dict, asset: str,
                   cov: dict) -> None:
    short = cfg["commit"][:7]
    src = Path(cfg["src_dir"])

    licenses = [f"Third-party licenses bundled with this whisper.cpp prebuilt ({cfg['tag']}).",
                f"Source: https://github.com/{cfg['source_repo']} @ {cfg['commit']}", ""]
    if (src / "LICENSE").is_file():
        shutil.copy2(src / "LICENSE", stage / "LICENSE")
        licenses += ["=== whisper.cpp LICENSE ===", (src / "LICENSE").read_text(), ""]
    for lic_dir in (src / "licenses", src / "LICENSES"):
        if lic_dir.is_dir():
            for lic in sorted(lic_dir.glob("*")):
                if lic.is_file():
                    licenses += [f"=== {lic.name} ===", lic.read_text(errors="replace"), ""]
    (stage / "THIRD_PARTY_LICENSES.txt").write_text("\n".join(licenses))

    info = {
        "asset_name": asset,
        "component": COMPONENT,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_tag": cfg["upstream_tag"],
        "packaging_tag": cfg["tag"],
        "source_repo": cfg["source_repo"],
        "source_repo_url": f"https://github.com/{cfg['source_repo']}",
        "source_commit": cfg["commit"],
        "source_commit_short": short,
        "os": strategy.name,
        "arch": cfg["arch"],
        "backend": cfg["backend"],
        "accel": cfg["accel"],
        "runtime_line": cov["runtime_line"],
        "coverage_class": cov["coverage_class"],
        "supported_sms": cov["supported_sms"],
        "min_sm": cov["min_sm"],
        "max_sm": cov["max_sm"],
        "gfx_target": cov["gfx_target"],
        "mapped_targets": cov["mapped_targets"],
        "min_os": cfg.get("min_os") or None,
        "static": cfg.get("static", False),
        "rpath": strategy.rpath,
        "studio_protocol": STUDIO_PROTOCOL,
    }
    if cfg.get("slim"):
        # The slim pairing contract: the installer may use this bundle only if
        # the paired llama.cpp install (same tag) provides these ggml sonames
        # plus the accelerator's backend module in its bin dir.
        info.update({
            "install_kind": "slim",
            "requires_llama_tag": cfg["llama_tag"],
            "requires_ggml_commit": paired_ggml_commit(cfg["llama_tag"]),
            "requires_ggml_version": cfg["ggml_version"],
            "requires_ggml_sonames": list(cfg.get("ggml_sonames") or strategy.ggml_sonames),
            "ggml_source": f"unslothai/llama.cpp@{cfg['llama_tag']}:/ggml",
        })
    (stage / INFO_NAME).write_text(json.dumps(info, indent=2))

    # First line is the Unsloth fingerprint. whisper.cpp does not compile a
    # build-info string into whisper-server (unlike llama.cpp's LLAMA_BUILD_TARGET),
    # so the brand cannot be baked into the binary; it is carried here in every
    # bundle instead. The assemble job's "Verify Unsloth fingerprint" gate greps
    # each bundle for this exact string, so keep it byte-identical to that MARK.
    build_info = [
        "Compiled by the Unsloth team",
        "",
        f"whisper.cpp upstream tag: {cfg['upstream_tag']}",
        f"packaging tag: {cfg['tag']}",
        f"os: {strategy.name}",
        f"arch: {cfg['arch']}",
        f"backend: {cfg['backend']}",
        f"accel: {cfg['accel']}",
        f"runtime line: {cov['runtime_line']}",
        f"coverage class: {cov['coverage_class']}",
        f"supported sms: {','.join(cov['supported_sms']) if cov['supported_sms'] else ''}",
        f"gfx target: {cov['gfx_target'] or ''}",
        f"min os: {cfg.get('min_os') or ''}",
        f"static: {'ON' if cfg.get('static') else 'OFF'}",
        f"rpath: {strategy.rpath}",
        f"source repo: {cfg['source_repo']}",
        f"source commit: {cfg['commit']}",
        f"source commit short: {short}",
        f"built at (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
    ]
    if cfg.get("slim"):
        build_info[-1:-1] = [
            "install kind: slim (ggml provided by the paired llama.cpp install)",
            f"paired llama tag: {cfg['llama_tag']}",
            f"ggml version: {cfg['ggml_version']}",
        ]
    (stage / "BUILD_INFO.txt").write_text("\n".join(build_info) + "\n")


def manifest_entry(info: dict, sha256: str) -> dict:
    """Per-asset manifest entry derived from a bundle's embedded info + its hash."""
    entry = {
        "asset": info["asset_name"],
        "os": info["os"],
        "arch": info["arch"],
        "backend": info["backend"],
        "accel": info["accel"],
        "install_kind": info.get("install_kind")
        or f"{info['os']}-{info['arch']}-{info['backend']}",
        "runtime_line": info.get("runtime_line"),
        "coverage_class": info.get("coverage_class"),
        "supported_sms": info.get("supported_sms"),
        "min_sm": info.get("min_sm"),
        "max_sm": info.get("max_sm"),
        "gfx_target": info.get("gfx_target"),
        "mapped_targets": info.get("mapped_targets"),
        "min_os": info.get("min_os"),
        "sha256": sha256,
    }
    if info.get("install_kind") == "slim":
        # Pairing requirements the installer checks before choosing slim. The
        # ggml commit is the ABI key; the installer pairs on it, not the tag.
        entry["requires_llama_tag"] = info.get("requires_llama_tag")
        entry["requires_ggml_commit"] = info.get("requires_ggml_commit")
        entry["requires_ggml_version"] = info.get("requires_ggml_version")
        entry["requires_ggml_sonames"] = info.get("requires_ggml_sonames")
    return entry


# --------------------------------------------------------------------------- #
# Package mode
# --------------------------------------------------------------------------- #
def do_package(args: argparse.Namespace) -> int:
    def env(k: str, default: str | None = None) -> str | None:
        return os.environ.get(k, default)

    cfg = {
        "bin_dir": args.bin_dir or env("BIN_DIR"),
        "src_dir": args.src_dir or env("SRC_DIR"),
        "out_dir": args.out_dir or env("OUT_DIR"),
        "tag": args.tag or env("TAG"),
        "upstream_tag": args.upstream_tag or env("UPSTREAM_TAG"),
        "commit": args.commit or env("SOURCE_COMMIT"),
        "source_repo": args.source_repo or env("SOURCE_REPO") or "unslothai/whisper.cpp",
        "os": args.os or env("OS") or "linux",
        "arch": args.arch or env("ARCH") or "x64",
        "accel": args.accel or env("ACCEL"),
        "backend": args.backend or env("BACKEND"),
        "runtime_line": args.runtime_line or env("RUNTIME_LINE") or "",
        "coverage_class": args.coverage_class or env("COVERAGE_CLASS") or "",
        "sms": args.sms or env("SMS") or "",
        "gfx_target": args.gfx_target or env("GFX_TARGET") or "",
        "mapped_targets": args.mapped_targets or env("MAPPED_TARGETS") or "",
        "min_os": args.min_os or env("MIN_OS") or "",
        "static": (args.static if args.static is not None
                   else (env("STATIC", "").lower() in ("1", "true", "on"))),
        "slim": (args.slim if args.slim is not None
                 else (env("SLIM", "").lower() in ("1", "true", "on"))),
        "llama_tag": args.llama_tag or env("LLAMA_TAG") or "",
        "ggml_version": args.ggml_version or env("GGML_VERSION") or "",
        # Optional override of the platform-default requires_ggml_sonames
        # (comma/space-separated). The macOS slim jobs use it per arch.
        "ggml_sonames": [s for s in re.split(r"[ ,;]+", args.ggml_sonames
                                             or env("GGML_SONAMES") or "") if s],
    }
    missing = [k for k in ("bin_dir", "src_dir", "out_dir", "tag", "upstream_tag",
                           "commit", "accel", "backend") if not cfg[k]]
    if missing:
        sys.exit(f"ERROR: missing required package inputs: {', '.join(missing)}")
    if cfg["slim"]:
        if not cfg["llama_tag"] or not cfg["ggml_version"]:
            sys.exit("ERROR: SLIM mode needs LLAMA_TAG and GGML_VERSION "
                     "(--llama-tag / --ggml-version)")
        if not re.fullmatch(r"\d+\.\d+\.\d+", cfg["ggml_version"]):
            sys.exit(f"ERROR: GGML_VERSION '{cfg['ggml_version']}' is not major.minor.patch")

    strategy = STRATEGIES.get(cfg["os"])
    if strategy is None:
        sys.exit(f"ERROR: unknown OS '{cfg['os']}' (have {sorted(STRATEGIES)})")

    bin_dir = Path(cfg["bin_dir"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cov = _coverage(cfg)
    asset = f"whisper-{cfg['tag']}-{strategy.name}-{cfg['arch']}-{cfg['accel']}{strategy.archive_ext}"

    stage = Path(tempfile.mkdtemp())
    try:
        if cfg["slim"]:
            curate_slim(strategy, bin_dir, stage)
        else:
            curate(strategy, bin_dir, stage)
        write_metadata(stage, strategy, cfg, asset, cov)
        out_path = out_dir / asset
        strategy.archive(stage, out_path)
        entry = manifest_entry(json.loads((stage / INFO_NAME).read_text()), sha256_file(out_path))
        (out_dir / f"{asset}.entry.json").write_text(json.dumps(entry, indent=2))
        print(f"wrote {out_path}")
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(stage).as_posix()}")
        print("manifest entry:")
        print(json.dumps(entry, indent=2))
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return 0


# --------------------------------------------------------------------------- #
# Aggregate mode
# --------------------------------------------------------------------------- #
def read_embedded_info(bundle: Path) -> dict:
    if bundle.name.endswith(".zip"):
        with zipfile.ZipFile(bundle) as zf:
            for n in zf.namelist():
                if n.endswith(INFO_NAME):
                    return json.loads(zf.read(n))
    else:
        with tarfile.open(bundle, "r:gz") as tar:
            for m in tar.getmembers():
                if m.isfile() and m.name.endswith(INFO_NAME):
                    return json.loads(tar.extractfile(m).read())
    sys.exit(f"ERROR: {bundle.name} has no {INFO_NAME}")


def do_aggregate(args: argparse.Namespace) -> int:
    dist = Path(args.dist)
    out = Path(args.out or args.dist)
    out.mkdir(parents=True, exist_ok=True)

    bundles = sorted(
        [p for p in list(dist.glob("whisper-*.tar.gz")) + list(dist.glob("whisper-*.zip"))
         if BUNDLE_RE.match(p.name)],
        key=lambda p: p.name,
    )
    if not bundles:
        sys.exit(f"ERROR: no whisper-*.{{tar.gz,zip}} bundles in {dist}")

    artifacts: list[dict] = []
    sha_index: dict[str, dict] = {}
    common = {
        "schema_version": 1,
        "component": COMPONENT,
        "paired_llama_tag": args.llama_tag or None,
        "paired_ggml_commit": paired_ggml_commit(args.llama_tag),
        "source_repo": args.source_repo,
        "source_repo_url": f"https://github.com/{args.source_repo}",
        "source_commit": args.commit,
        "source_commit_short": args.commit[:7],
        "upstream_repo": UPSTREAM_REPO,
        "upstream_tag": args.upstream_tag,
        "packaging_tag": args.tag,
        "studio_protocol": STUDIO_PROTOCOL,
    }

    for b in bundles:
        info = read_embedded_info(b)
        digest = sha256_file(b)
        artifacts.append(manifest_entry(info, digest))
        sha_index[b.name] = {
            "kind": f"{info['os']}-{info['arch']}-{info['backend']}-bundle",
            "repo": args.source_repo,
            "sha256": digest,
            "source_commit": args.commit,
            "upstream_tag": args.upstream_tag,
        }

    # Source archives (whisper.cpp-source-*.tar.gz): hash whatever is present.
    for p in sorted(dist.glob("whisper.cpp-source-*.tar.gz")):
        sha_index[p.name] = {
            "kind": "source",
            "repo": args.source_repo,
            "sha256": sha256_file(p),
            "source_commit": args.commit,
            "upstream_tag": args.upstream_tag,
        }

    manifest = {
        **common,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifacts": artifacts,
    }
    if args.emit_manifest:
        mp = out / "whisper-prebuilt-manifest.json"
        mp.write_text(json.dumps(manifest, indent=2))
        sha_index["whisper-prebuilt-manifest.json"] = {
            "kind": "published-manifest", "repo": args.source_repo,
            "sha256": sha256_file(mp), "source_commit": args.commit,
            "upstream_tag": args.upstream_tag,
        }
        print(f"wrote {mp} ({len(artifacts)} artifacts)")

    if args.emit_sha256:
        sp = out / "whisper-prebuilt-sha256.json"
        sp.write_text(json.dumps({**common, "release_tag": args.tag,
                                  "artifacts": sha_index}, indent=2))
        print(f"wrote {sp} ({len(sha_index)} entries)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Package + manifest tool for Unsloth whisper.cpp prebuilts")
    ap.add_argument("--emit-manifest", action="store_true", help="aggregate mode: write whisper-prebuilt-manifest.json")
    ap.add_argument("--emit-sha256", action="store_true", help="aggregate mode: write whisper-prebuilt-sha256.json")
    ap.add_argument("--dist", help="aggregate mode: dir holding the built bundles")
    ap.add_argument("--out", help="aggregate mode: dir to write the sidecars (default --dist)")
    # package-mode inputs (each also falls back to the matching env var)
    ap.add_argument("--bin-dir")
    ap.add_argument("--src-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--tag", help="packaging tag, e.g. v1.9.1-unsloth.1")
    ap.add_argument("--upstream-tag", help="upstream whisper.cpp tag, e.g. v1.9.1")
    ap.add_argument("--commit")
    ap.add_argument("--source-repo")
    ap.add_argument("--os", choices=["linux", "macos", "windows"])
    ap.add_argument("--arch", choices=["x64", "arm64"])
    ap.add_argument("--accel", help="cpu | cuda12-portable | vulkan | rocm-gfx908 | metal | slim ...")
    ap.add_argument("--backend", choices=["cpu", "cuda", "vulkan", "rocm", "metal", "slim"])
    ap.add_argument("--runtime-line")
    ap.add_argument("--coverage-class")
    ap.add_argument("--sms")
    ap.add_argument("--gfx-target")
    ap.add_argument("--mapped-targets")
    ap.add_argument("--min-os")
    ap.add_argument("--static", dest="static", action="store_true", default=None)
    # slim mode (package): only whisper-server + libwhisper ship; ggml comes
    # from the paired llama.cpp release at runtime. --llama-tag doubles as the
    # aggregate-mode source of the manifest's top-level paired_llama_tag.
    ap.add_argument("--slim", dest="slim", action="store_true", default=None)
    ap.add_argument("--llama-tag", help="paired unslothai/llama.cpp release tag")
    ap.add_argument("--ggml-version", help="ggml version compiled against (major.minor.patch)")
    ap.add_argument("--ggml-sonames", help="override requires_ggml_sonames "
                    "(comma/space-separated library filenames)")
    args = ap.parse_args()

    if args.emit_manifest or args.emit_sha256:
        for req in ("dist", "tag", "upstream_tag", "commit", "source_repo"):
            if not getattr(args, req):
                sys.exit(f"ERROR: aggregate mode needs --{req.replace('_', '-')}")
        return do_aggregate(args)
    return do_package(args)


if __name__ == "__main__":
    raise SystemExit(main())
