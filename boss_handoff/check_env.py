#!/usr/bin/env python3
"""
Swarm Scan -- environment checker (read-only, no install, no sudo).

Run this on any machine to find out whether the Swarm Scan Blender addon
will run there, and whether the GPU can be used for fast rendering.

    python3 check_env.py

It only *reads* system state. It never installs or changes anything.
"""

import os
import platform
import re
import shutil
import subprocess
import sys

TARGET_BLENDER = (5, 2, 0)          # the version this project targets
MIN_BLENDER = (4, 2, 0)             # the addon's declared minimum (bl_info)

# ---- tiny output helpers ---------------------------------------------------
def line(): print("-" * 64)
def head(t): line(); print(t); line()
def ok(t): print(f"  [OK]    {t}")
def warn(t): print(f"  [!]     {t}")
def bad(t): print(f"  [X]     {t}")
def info(t): print(f"          {t}")


def run(cmd, timeout=25):
    """Run a command, return (rc, stdout+stderr). Never raises."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return -1, "(timed out)"
    except Exception as e:  # noqa: BLE001
        return -2, str(e)


# ---- 1. distro / OS --------------------------------------------------------
def check_os():
    head("1. Operating system")
    system = platform.system()
    if system == "Linux":
        pretty = None
        try:
            with open("/etc/os-release") as f:
                for ln in f:
                    if ln.startswith("PRETTY_NAME="):
                        pretty = ln.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        info(f"OS: {pretty or 'Linux (unknown distro)'}")
        info(f"Kernel: {platform.release()}  ({platform.machine()})")
    elif system == "Darwin":
        info(f"OS: macOS {platform.mac_ver()[0]}  ({platform.machine()})")
    else:
        info(f"OS: {system} {platform.release()}  ({platform.machine()})")
    return system


# ---- 2. locate Blender -----------------------------------------------------
def find_blender():
    # env override wins
    env = os.environ.get("SWARM_BLENDER")
    if env and os.path.exists(env):
        return env
    on_path = shutil.which("blender")
    if on_path:
        return on_path
    candidates = [
        "/opt/blender/blender",
        "/usr/local/bin/blender",
        "/snap/bin/blender",
        os.path.expanduser("~/blender/blender"),
        os.path.expanduser("~/Downloads/blender/blender"),
        # macOS fallbacks so this stays cross-platform
        "/Applications/Blender.app/Contents/MacOS/Blender",
        os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def parse_version(text):
    m = re.search(r"Blender\s+(\d+)\.(\d+)\.(\d+)", text)
    return tuple(int(x) for x in m.groups()) if m else None


def check_blender():
    head("2. Blender")
    exe = find_blender()
    if not exe:
        bad("Blender not found (checked PATH, /opt/blender, /usr/local/bin, "
            "~/blender, Downloads, macOS app locations).")
        info("You need to install it -- see README_BOSS.md, step 1.")
        return None, None
    rc, out = run([exe, "--version"])
    ver = parse_version(out)
    ok(f"Found: {exe}")
    if not ver:
        warn("Could not read version from `blender --version` output.")
        return exe, None
    vstr = ".".join(map(str, ver))
    if ver == TARGET_BLENDER:
        ok(f"Version {vstr} -- matches target {'.'.join(map(str, TARGET_BLENDER))}.")
    elif ver >= MIN_BLENDER:
        warn(f"Version {vstr} -- not the {'.'.join(map(str, TARGET_BLENDER))} target, "
             f"but >= {'.'.join(map(str, MIN_BLENDER))} so the addon should load.")
    else:
        bad(f"Version {vstr} -- older than the addon minimum "
            f"{'.'.join(map(str, MIN_BLENDER))}. Install a newer Blender.")
    return exe, ver


# ---- 3. GPU hardware + NVIDIA driver health --------------------------------
def check_gpu_hardware(system):
    head("3. Graphics hardware")
    gpu_lines = []
    if system == "Linux":
        rc, out = run(["lspci"])
        if rc == 0:
            gpu_lines = [ln.strip() for ln in out.splitlines()
                         if re.search(r"VGA|3D|Display", ln, re.I)]
        for g in gpu_lines:
            info(g)
        if not gpu_lines:
            warn("Could not enumerate GPUs via lspci.")
    elif system == "Darwin":
        rc, out = run(["system_profiler", "SPDisplaysDataType"])
        for ln in out.splitlines():
            if "Chipset Model" in ln or "Vendor" in ln:
                info(ln.strip())
    else:
        info("GPU enumeration not implemented for this OS; relying on Blender check below.")

    # NVIDIA driver health -- the common "card present, driver missing/broken" case
    nvidia_present = any("nvidia" in g.lower() for g in gpu_lines)
    smi = shutil.which("nvidia-smi")
    nvidia_ok = False
    if smi:
        rc, out = run([smi, "--query-gpu=name,driver_version",
                       "--format=csv,noheader"])
        if rc == 0 and out.strip():
            ok(f"nvidia-smi works: {out.strip().splitlines()[0]}")
            nvidia_ok = True
        else:
            bad("nvidia-smi is installed but FAILED to run -- driver is likely "
                "broken or the kernel module isn't loaded. GPU rendering won't work "
                "until the NVIDIA driver is fixed (README step 'If GPU isn't detected').")
    else:
        if nvidia_present:
            bad("NVIDIA card detected but nvidia-smi is NOT installed -- the driver "
                "is missing. Install the NVIDIA driver (README step 'If GPU isn't "
                "detected') to use the GPU.")
        else:
            info("No NVIDIA tooling found (only relevant if this machine has an "
                 "NVIDIA card).")
    return nvidia_present, nvidia_ok


# ---- 4. Cycles compute devices (the real test -- runs inside Blender) ------
CYCLES_PROBE = r"""
import bpy
try:
    prefs = bpy.context.preferences.addons['cycles'].preferences
    backends = [t[0] for t in prefs.get_device_types(bpy.context)]
    print('SWARM_BACKENDS:' + ','.join(backends))
    for be in backends:
        if be in ('NONE', 'CPU'):
            continue
        try:
            prefs.compute_device_type = be
            prefs.get_devices()
            gpus = [d.name for d in prefs.devices if d.type == be]
            if gpus:
                print('SWARM_GPU:' + be + ':' + ' | '.join(gpus))
        except Exception as e:
            print('SWARM_ERR:' + be + ':' + str(e))
except Exception as e:
    print('SWARM_FATAL:' + str(e))
"""


def check_cycles(exe, nvidia_present, nvidia_ok):
    head("4. Can Cycles use the GPU?")
    if not exe:
        warn("Skipped -- Blender isn't installed yet. Install it, then re-run.")
        return None
    rc, out = run([exe, "-b", "--factory-startup", "--python-expr", CYCLES_PROBE],
                  timeout=90)
    backends, gpus = [], []
    for ln in out.splitlines():
        if ln.startswith("SWARM_BACKENDS:"):
            backends = [b for b in ln.split(":", 1)[1].split(",") if b]
        elif ln.startswith("SWARM_GPU:"):
            _, be, names = ln.split(":", 2)
            gpus.append((be, names))
        elif ln.startswith("SWARM_FATAL:"):
            bad("Could not query Cycles: " + ln.split(":", 1)[1])
    if not backends:
        warn("Blender ran but returned no Cycles device info "
             "(unexpected -- try re-running).")
        return None
    # NOTE: get_device_types() always lists CUDA/OPTIX/HIP/ONEAPI (the backends
    # this Blender build was compiled with) regardless of hardware. Only the
    # presence of an actual enumerated GPU *device* means the GPU is usable.
    if gpus:
        for be, names in gpus:
            ok(f"Cycles can use the GPU: {be} -> {names}")
        info("GPU rendering WILL be used. This is the fast path.")
        return True
    # No usable GPU device. Distinguish "broken driver" from "no discrete GPU".
    if nvidia_present and not nvidia_ok:
        bad("An NVIDIA card is present but Cycles found NO usable GPU device -- "
            "the driver is missing or broken. Fix the NVIDIA driver (README step "
            "'If the checker says the GPU isn't usable') to render on the GPU.")
    else:
        info("No compatible GPU for Cycles on this machine -- normal for "
             "integrated graphics or a box with no discrete NVIDIA/AMD card.")
    info("Cycles will render on the CPU: slower, but it works fine for "
         "messing around.")
    return False


# ---- 5. disk space ---------------------------------------------------------
def check_disk():
    head("5. Free disk space")
    try:
        total, used, free = shutil.disk_usage(os.path.expanduser("~"))
        gb = free / (1024 ** 3)
        (ok if gb >= 5 else warn)(f"{gb:.0f} GB free in your home folder.")
        if gb < 5:
            info("Under 5 GB free -- Blender + a few renders may run tight.")
    except Exception as e:  # noqa: BLE001
        warn(f"Could not read disk space: {e}")


# ---- verdict ---------------------------------------------------------------
def verdict(blender_ver, cycles_gpu):
    head("VERDICT")
    if blender_ver is None:
        print("  >> NOT READY: install Blender first (README_BOSS.md, step 1),")
        print("     then run this checker again.")
        return
    if blender_ver < MIN_BLENDER:
        print("  >> NOT READY: your Blender is too old. Install a newer one")
        print("     (README_BOSS.md, step 1), then re-run this checker.")
        return
    if cycles_gpu is True:
        print("  >> YOU'RE READY. GPU rendering is available (the fast path).")
    elif cycles_gpu is False:
        print("  >> YOU'RE READY (with CPU rendering). The GPU isn't usable, so")
        print("     renders will be slower -- fine for trying things out. To speed")
        print("     it up later, fix the GPU driver (README 'If GPU isn't detected').")
    else:
        print("  >> ALMOST READY. Blender is installed but the GPU check didn't")
        print("     complete. You can still launch the addon (README 'Launch the addon').")
    print()
    print("  Next: follow README_BOSS.md 'Launch the addon'.")


def main():
    print("\n==== Swarm Scan environment check ====")
    print("(read-only: this does not install or change anything)\n")
    system = check_os()
    exe, ver = check_blender()
    nvidia_present, nvidia_ok = check_gpu_hardware(system)
    cycles_gpu = check_cycles(exe, nvidia_present, nvidia_ok)
    check_disk()
    verdict(ver, cycles_gpu)
    print()


if __name__ == "__main__":
    main()
