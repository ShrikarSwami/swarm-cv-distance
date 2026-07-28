# Swarm Scan — Linux setup (the easy way)

Three commands. Copy each line, paste it into a terminal, press Enter.
When you're done, Blender opens with the tool already loaded — every time.

> **Open a terminal:** press the **Super/Windows key**, type `terminal`,
> press Enter.

---

## Step 1 — Go into the folder

You were given a file **`swarm-scan-boss.zip`** (it's in your Downloads).
Unzip it and go inside:

```bash
cd ~/Downloads && unzip -o swarm-scan-boss.zip && cd swarm-scan-boss
```

**You should see:** your terminal prompt now ends in `swarm-scan-boss`.

---

## Step 2 — Run the setup (this installs everything)

```bash
./setup_linux.sh
```

Wait until it prints **`DONE. Everything is installed.`**
This downloads Blender, sets it up, and makes the tool load automatically.
It never asks for a password and doesn't change anything else on your computer.

---

## Step 3 — Open it

```bash
~/blender/blender
```

(Or, from now on, just click **"Blender (Swarm Scan)"** in your applications
menu.)

**You should see:** Blender opens in a window. The tool is **already loaded** —
you do **not** have to install or enable anything.

---

## Use it

Inside the Blender window:

1. Move the mouse over the big 3D area and press the **`N`** key.
   A sidebar opens on the right. Click the **`Swarm Scan`** tab.
2. Click **`Generate Swarm`** — a cloud of drones appears.
3. Click **`Start Flight Sim`** — they fly like a flock (`Stop Flight Sim` to freeze).
4. Click **`Place Cameras`** — a ring of cameras appears around them.

To look around: hold the **middle mouse button** and drag to rotate; scroll to zoom.

That's everything. To open it again later, just do Step 3.

---

## Optional: is my graphics card being used?

Renders are faster with a supported NVIDIA card. To check:

```bash
cd ~/Downloads/swarm-scan-boss && python3 check_env.py
```

Read the last line (starts with `>>`):

- `>> YOU'RE READY.` — the GPU is being used. 
- `>> YOU'RE READY (with CPU rendering).` — no usable GPU, so it runs on the
  CPU. **Slower, but works fine** — you can ignore this.

If it says an **NVIDIA driver is missing/broken**, install the driver, then
**restart the computer**:

```bash
sudo ubuntu-drivers autoinstall
```

(That's the only command here that needs your password. If you don't have an
NVIDIA card, skip this — CPU rendering is fine for messing around.)
