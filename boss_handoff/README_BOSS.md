# Swarm Scan — Setup

This runs the interactive drone-swarm tool inside Blender: generate a swarm,
fly it around (boids), place a camera rig, and scan it.

Follow the steps in order. **Copy each command, paste it into a terminal,
press Enter, and check you see what's described.** You don't need to
understand the commands.

> Open a terminal: on most Linux desktops, press the **Super/Windows key**,
> type `terminal`, and press Enter.

---

## 1. Install Blender

Copy-paste these four lines one block at a time:

```bash
cd ~
wget https://download.blender.org/release/Blender5.2/blender-5.2.0-linux-x64.tar.xz
tar -xf blender-5.2.0-linux-x64.tar.xz
mv blender-5.2.0-linux-x64 blender
```

Check it worked:

```bash
~/blender/blender --version
```

**You should see:** a line starting with `Blender 5.2.0`.

> Don't use "apt", "snap", or "flatpak" to install Blender — those often give
> an old version that won't match this tool. The download above is the
> official one and is the version you want.

---

## 2. Get the project files

You were sent a file called **`swarm-scan-boss.zip`**. Put it in your
`Downloads` folder, then run:

```bash
cd ~
unzip ~/Downloads/swarm-scan-boss.zip
cd ~/swarm-scan-boss
```

**You should see:** no error, and your terminal prompt now ends in
`swarm-scan-boss`.

---

## 3. Run the checker

This tells you, in plain English, whether everything is ready. It only looks
at your system — it installs nothing.

```bash
python3 check_env.py
```

**You should see:** a report ending with a line that starts with `>>`. Read
that last line:

- `>> YOU'RE READY.` — great, go to step 5.
- `>> YOU'RE READY (with CPU rendering).` — also fine. The tool works; it's
  just a bit slower. Go to step 5.
- `>> NOT READY: install Blender first` — step 1 didn't finish. Redo step 1.

> If you see `python3: command not found`, use Blender's copy instead:
> ```bash
> ~/blender/blender -b --python check_env.py
> ```

---

## 4. If the checker says the GPU isn't usable

This is only relevant if the machine has an **NVIDIA** graphics card and the
checker reported a missing/broken driver.

On Ubuntu / Linux Mint / Pop!_OS, install the driver:

```bash
sudo ubuntu-drivers autoinstall
```

Then **restart the computer**, and run the checker again (step 3):

```bash
cd ~/swarm-scan-boss && python3 check_env.py
```

**You should see:** the checker now reports the NVIDIA card and `YOU'RE READY`.

> If it still doesn't work, don't worry — **CPU rendering works fine for
> messing around**, just slower. You can keep going with step 5.

---

## 5. Launch the addon

```bash
cd ~/swarm-scan-boss
./launch.sh
```

**You should see:** Blender opens in a window. (The terminal will print
`Using Blender: ...` and `Swarm Scan addon registered`.)

> If it says `could not find Blender`, run this instead:
> ```bash
> SWARM_BLENDER=~/blender/blender ./launch.sh
> ```

---

## 6. First things to try (the fun part)

Inside the Blender window:

1. **Open the tool panel.** Move your mouse over the big 3D area and press
   the **`N`** key. A sidebar appears on the right. Click the **`Swarm Scan`**
   tab in that sidebar.

2. **Make a swarm.** Click **`Generate Swarm`**. A cloud of drones appears in
   the 3D view.

3. **Fly it.** Click **`Start Flight Sim`**. The drones start moving like a
   flock. Click **`Stop Flight Sim`** to freeze them.

4. **Add cameras.** Click **`Place Cameras`**. A ring of cameras appears
   around the swarm.

To look around: hold the **middle mouse button** and drag to rotate the view;
scroll the wheel to zoom.

> The **`Run Scan`** button is an advanced feature that needs extra setup and
> a full render — skip it for now; the sim and camera rig above are the parts
> to play with first.

---

That's it. To open it again later, just repeat step 5.
