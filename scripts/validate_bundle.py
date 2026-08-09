#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Validate an Unsloth whisper.cpp prebuilt bundle before it is published.

Given an extracted bundle directory, a tiny whisper GGML model and an audio
clip, this asserts the bundle is actually runnable and speaks the Studio
/inference contract:

  1. whisper-server --help returns 0 (the binary loads its own libs).
  2. dependency-closure check (ldd / otool -L): no unresolved ("not found")
     libraries. Skipped on Windows (no portable equivalent).
  3. start whisper-server with the model, POST one real multipart /inference
     request, assert HTTP 200 and a non-empty JSON `text`.
  4. repeat step 3 with --no-gpu (Studio appends it during training), so a GPU
     bundle is proven to also fall back to CPU.

--gpu marks a run as GPU inference (fails if the server reports no device);
otherwise the server is launched normally and CPU is fine. Real GPU inference is
meant for a self-hosted accelerator runner (see the workflow / README).

--ggml-dir <dir> validates a SLIM bundle (which ships no libggml*): every ggml
object from that directory (an extracted paired unslothai/llama.cpp bundle),
plus the OpenMP runtime the clang-built slices link it against, is linked into
the bundle dir first -- exactly the wiring the Studio installer performs -- and
then the same checks run. ggml's registry scans the executable's directory for
backend modules, so links in the bin dir are the only wiring that registers both
the CPU and accelerator backends. A fifth check then asserts that every library
the loader took from the llama dir is published in the bundle's
requires_ggml_sonames, so CI's wiring and the installer's cannot drift apart.

Wiring only ever touches the EXTRACTED bundle, never the packaged archive, and
skips names already present, so re-running is a no-op on an already wired dir.

Exit 0 = valid; nonzero = do not ship this asset.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_C_ENV = {**os.environ, "LC_ALL": "C", "LANG": "C"}

# Metadata package_bundle.py embeds in every bundle; carries the slim pairing
# contract (requires_ggml_sonames) the installer wires from.
INFO_NAME = "UNSLOTH_WHISPER_PREBUILT_INFO.json"

# Libraries the host provides at runtime, so an unresolved reference to them is
# expected on a GPU-less build runner and is NOT a packaging defect: the NVIDIA
# driver (libcuda), the CUDA runtime that package_bundle.py deliberately does not
# bundle (paired with the user's PyTorch), and the Vulkan loader ICD. They resolve
# on a real accelerator host. A genuinely missing BUNDLED lib (libggml*, libwhisper,
# a co-located backend module) is not on this list and still fails the check.
_HOST_PROVIDED_LIB = re.compile(
    r"^lib(cuda|cudart|cublas(lt)?|cufft|curand|cusparse|cusolver|nvrtc(-builtins)?|"
    r"nvjitlink|nvidia-[a-z0-9_.+-]+|vulkan)\.so",
    re.I,
)


# The ggml objects themselves: the libraries whisper-server links against plus
# the dlopen'd backend modules ggml's registry scans the bin dir for.
_GGML_PATTERNS = ("libggml*.so", "libggml*.so.*", "libggml*.dylib", "ggml*.dll")

# The OpenMP runtime rides along with them. It is NOT a libggml* name and NOT a
# system library: llama's clang-built slices link ggml-base (and the CPU
# variants) against the LLVM runtime and ship it inside the bundle, so it has to
# land next to whisper-server too or the loader cannot bring ggml up at all:
#   windows x64/arm64  ggml-base.dll imports libomp140.<arch>.dll
#   linux-arm64        libggml-base.so NEEDS libomp.so.5 (clang 19, RUNPATH $ORIGIN)
# Slices built with gcc (linux-x64, whose libgomp.so.1 comes from the host, as
# it does for llama's own binaries) or Apple clang (macOS, no OpenMP) ship no
# libomp at all, so the patterns simply do not match there.
_OPENMP_PATTERNS = ("libomp*.so", "libomp*.so.*", "libomp*.dylib", "libomp*.dll")


def wire_ggml(bundle: Path, ggml_dir: Path) -> set[str]:
    """Link every ggml object from --ggml-dir into the bundle dir (slim wiring).

    Symlink where the platform allows it, copy otherwise (Windows). Existing
    names in the bundle are left alone -- though a slim bundle ships none, by
    packaging contract -- so re-running over an already wired bundle is a no-op
    and leaves byte-identical contents. Only the EXTRACTED copy is ever touched;
    the packaged archive is never rewritten.

    Returns the set of names this directory provides (whether linked now or
    already present), i.e. the wiring contract the installer has to reproduce.
    """
    if not ggml_dir.is_dir():
        sys.exit(f"ERROR: --ggml-dir is not a directory: {ggml_dir}")
    names: set[str] = set()
    for pat in _GGML_PATTERNS + _OPENMP_PATTERNS:
        names.update(p.name for p in ggml_dir.glob(pat))
    if not any(n.startswith(("libggml", "ggml")) for n in names):
        sys.exit(f"ERROR: no ggml libraries (libggml* / ggml*.dll) in {ggml_dir}")
    wired = 0
    for name in sorted(names):
        src, dst = ggml_dir / name, bundle / name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            os.symlink(os.path.abspath(src), dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst, follow_symlinks=True)
        wired += 1
    print(f"OK: wired {wired} ggml objects from {ggml_dir} into {bundle} "
          f"({len(names)} provided)")
    return names


def _server_path(bundle: Path) -> Path:
    for name in ("whisper-server", "whisper-server.exe"):
        p = bundle / name
        if p.exists():
            return p
    sys.exit(f"ERROR: no whisper-server(.exe) in {bundle}")


def _child_env(bundle: Path) -> dict:
    """Prepend the bundle dir to the loader path so co-located libs resolve."""
    env = dict(os.environ)
    key = {"Linux": "LD_LIBRARY_PATH", "Darwin": "DYLD_LIBRARY_PATH"}.get(platform.system())
    if key:
        env[key] = os.pathsep.join([str(bundle), env.get(key, "")]).rstrip(os.pathsep)
    env["PATH"] = os.pathsep.join([str(bundle), env.get("PATH", "")])
    return env


def check_help(server: Path, env: dict) -> None:
    r = subprocess.run([str(server), "--help"], capture_output=True, text=True,
                       env=env, timeout=60)
    # whisper-server prints usage and may exit 0 or non-zero for --help across
    # versions; treat "usage/options text present" as success and only fail on a
    # loader error (missing library) which yields no usage text.
    combined = (r.stdout + r.stderr).lower()
    if "usage" not in combined and "options" not in combined and "--model" not in combined:
        sys.exit(f"ERROR: whisper-server --help produced no usage text (rc={r.returncode}):\n{combined[:2000]}")
    print("OK: whisper-server --help")


def check_wiring_contract(bundle: Path, server: Path, ggml_dir: Path,
                          provided: set[str]) -> None:
    """Every lib the loader pulls out of the paired llama bundle must be in
    requires_ggml_sonames.

    CI wires the bundle by globbing the llama dir, but the Studio installer wires
    it from the manifest's requires_ggml_sonames list. If those two disagree the
    build goes green while every install is broken -- which is exactly how
    linux/arm64 shipped a whisper-server that died on a missing libomp.so.5.
    Linux only: ldd is the only portable way to read the transitive NEEDED set
    (macOS has its own @rpath gate in the workflow, Windows has no equivalent).
    """
    if platform.system() != "Linux":
        print("SKIP: wiring-contract check (ldd is Linux-only)")
        return
    info_path = bundle / INFO_NAME
    if not info_path.is_file():
        print(f"SKIP: wiring-contract check ({INFO_NAME} not in bundle)")
        return
    info = json.loads(info_path.read_text())
    declared = set(info.get("requires_ggml_sonames") or [])
    if not declared:
        print("SKIP: wiring-contract check (bundle declares no requires_ggml_sonames)")
        return
    r = subprocess.run(["ldd", str(server)], capture_output=True, text=True,
                       env=_child_env(bundle))
    # Left of "=>" is the soname the loader asked for, resolved or not. A soname
    # the llama dir provides is by definition satisfied by the wiring, never by
    # the host, so it belongs in the contract.
    needed = {line.split("=>")[0].strip() for line in r.stdout.splitlines() if "=>" in line}
    missing = sorted(needed & provided - declared)
    if missing:
        sys.exit(
            "ERROR: whisper-server loads these libraries out of the paired llama "
            f"bundle but the manifest does not require them: {', '.join(missing)}\n"
            f"       declared requires_ggml_sonames: {', '.join(sorted(declared))}\n"
            "       add them to the platform strategy's sonames in "
            "package_bundle.py (or GGML_SONAMES for this job), or the installer "
            "will wire an unloadable bundle.")
    absent = sorted(declared - provided)
    if absent:
        print(f"WARNING: requires_ggml_sonames names {', '.join(absent)}, absent from "
              f"{ggml_dir}; the installer's pairing check would reject this bundle",
              file=sys.stderr)
    print(f"OK: wiring contract ({len(needed & provided)} llama-provided libs, all declared)")


def check_closure(bundle: Path, server: Path) -> None:
    system = platform.system()
    env = _child_env(bundle)
    if system == "Windows":
        print("SKIP: dependency-closure check not available on Windows")
        return
    if system == "Darwin":
        # otool cannot report unresolved refs directly, so load-test the server:
        # a missing @rpath/@loader_path dependency makes dyld abort at launch
        # with "Library not loaded" / "image not found".
        r = subprocess.run([str(server), "--help"], capture_output=True, text=True,
                           env=env, timeout=60)
        blob = (r.stdout + r.stderr).lower()
        if "library not loaded" in blob or "image not found" in blob:
            sys.exit(f"ERROR: dyld could not resolve the bundle's libraries:\n{r.stderr[:2000]}")
        print("OK: dependency closure (dyld loaded the bundle cleanly)")
        return
    # Linux: ldd every object, flag any "not found" that is not a host-provided
    # driver/runtime lib (GPU bundles resolve those on a real accelerator host).
    targets = [server] + sorted(list(bundle.glob("*.so")) + list(bundle.glob("*.so.*")))
    unresolved: list[str] = []
    external: set[str] = set()
    for t in targets:
        r = subprocess.run(["ldd", str(t)], capture_output=True, text=True, env=env)
        for line in r.stdout.splitlines():
            if "not found" not in line:
                continue
            soname = line.strip().split()[0]
            if _HOST_PROVIDED_LIB.match(soname):
                external.add(soname)
            else:
                unresolved.append(f"{t.name}: {line.strip()}")
    if unresolved:
        sys.exit("ERROR: unresolved libraries in bundle:\n" + "\n".join(unresolved))
    if external:
        print("NOTE: host-provided libs, resolved on a real accelerator host, "
              "not bundled by design: " + ", ".join(sorted(external)))
    print(f"OK: dependency closure ({len(targets)} objects, no unresolved bundled libs)")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_inference(server: Path, model: Path, audio: Path, bundle: Path, *,
                  no_gpu: bool, require_gpu: bool) -> None:
    port = _free_port()
    cmd = [str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port)]
    if no_gpu:
        cmd.append("--no-gpu")
    env = _child_env(bundle)
    mode = "no-gpu" if no_gpu else "gpu" if require_gpu else "default"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env)
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 90
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                sys.exit(f"ERROR: whisper-server exited early (rc={proc.returncode}) [{mode}]:\n{out[:3000]}")
            try:
                urllib.request.urlopen(base + "/", timeout=2)
                ready = True
                break
            except urllib.error.HTTPError:
                # Any HTTP status (e.g. 404 on /) means the server is listening.
                ready = True
                break
            except Exception:
                time.sleep(1)
        if not ready:
            sys.exit(f"ERROR: whisper-server did not become ready in time [{mode}]")

        boundary = "----unslothvalidate"
        parts = []
        for field, value in (("temperature", "0.0"), ("response_format", "json"),
                             ("beam_size", "1"), ("language", "en")):
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"\r\n\r\n{value}\r\n")
        head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                f"filename=\"{audio.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n")
        body = b"".join(p.encode() for p in parts) + head.encode() + audio.read_bytes() \
            + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(base + "/inference", data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status != 200:
                sys.exit(f"ERROR: /inference returned HTTP {resp.status} [{mode}]")
            payload = json.loads(resp.read())
        text = (payload.get("text") or "").strip()
        if not text:
            sys.exit(f"ERROR: /inference returned empty text [{mode}]: {payload}")
        print(f"OK: /inference [{mode}] transcript = {text[:80]!r}")

        if require_gpu:
            # A GPU run must have actually used a device; whisper-server logs the
            # backend it selected. Absence of a CUDA/Metal/ROCm/Vulkan device
            # line means it silently ran on CPU, which does not validate the GPU.
            # Terminate first so the stdout pipe reaches EOF: reading a live
            # server's stdout to EOF never returns and would hang the job.
            proc.terminate()
            try:
                log = proc.communicate(timeout = 15)[0] or ""
            except subprocess.TimeoutExpired:
                proc.kill()
                log = proc.communicate()[0] or ""
            if not re.search(r"(CUDA|Metal|ROCm|HIP|Vulkan|GPU)", log, re.I):
                print("WARNING: could not confirm a GPU device from server log; "
                      "inspect the accelerator runner output", file=sys.stderr)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path, help="extracted bundle directory")
    ap.add_argument("--model", required=True, type=Path, help="tiny whisper GGML model (.bin)")
    ap.add_argument("--audio", required=True, type=Path, help="audio clip (e.g. samples/jfk.wav)")
    ap.add_argument("--gpu", action="store_true", help="require real GPU inference (accelerator runner)")
    ap.add_argument("--cpu-only", action="store_true",
                    help="GPU-less build runner: run help + closure + one --no-gpu transcription only")
    ap.add_argument("--skip-no-gpu", action="store_true", help="skip the --no-gpu fallback run")
    ap.add_argument("--ggml-dir", type=Path,
                    help="slim bundles: link every ggml object from this extracted "
                         "paired llama.cpp bundle into the bundle dir before checking")
    args = ap.parse_args()

    for p in (args.bundle, args.model, args.audio):
        if not p.exists():
            sys.exit(f"ERROR: not found: {p}")

    provided: set[str] = set()
    if args.ggml_dir:
        provided = wire_ggml(args.bundle, args.ggml_dir)

    server = _server_path(args.bundle)
    if os.name != "nt":
        os.chmod(server, 0o755)
    env = _child_env(args.bundle)

    check_help(server, env)
    check_closure(args.bundle, server)
    if args.ggml_dir:
        check_wiring_contract(args.bundle, server, args.ggml_dir, provided)
    if args.cpu_only:
        # GPU-less runner: the only meaningful path is the CPU fallback that
        # Studio drives with --no-gpu. Real GPU inference is a separate job.
        run_inference(server, args.model, args.audio, args.bundle, no_gpu=True, require_gpu=False)
        print("VALID: bundle passed CPU-only checks")
        return 0
    # Default / GPU run.
    run_inference(server, args.model, args.audio, args.bundle, no_gpu=False, require_gpu=args.gpu)
    # CPU-fallback run (Studio appends --no-gpu during training).
    if not args.skip_no_gpu:
        run_inference(server, args.model, args.audio, args.bundle, no_gpu=True, require_gpu=False)
    print("VALID: bundle passed all checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
