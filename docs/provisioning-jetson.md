# Jetson Provisioning Guide

This guide takes a fresh NVIDIA Jetson Orin Nano from an unboxed board to
a running `kestrel` REPL. Steps that cannot be fully verified without
physical hardware are flagged inline; treat them as a checklist to
confirm on your own device, not as a guarantee.

## Prerequisites

- **Board:** Jetson Orin Nano 8GB (Super), its official power supply, and
  a USB cable capable of putting it into recovery mode for flashing.
- **Storage:** An NVMe SSD (M.2 form factor compatible with the Orin Nano
  Developer Kit carrier board). The board's onboard storage alone is too
  small for Kestrel's repos, virtual environments, and session logs.
- **Host PC:** A separate Ubuntu x86_64 machine to run NVIDIA SDK
  Manager -- SDK Manager does not run on the Jetson itself, and does not
  ship for Windows or macOS.
- **Accounts:** A free NVIDIA Developer account (required to download
  JetPack images through SDK Manager), and an API key for at least one
  model backend -- an OpenRouter key, a Z.ai key, or both.

## Flash JetPack 6.2

`[UNVERIFIED]` The steps below reflect NVIDIA's published flashing
procedure; they have not yet been run end-to-end against a physical
board as part of this guide.

1. Install NVIDIA SDK Manager on the host PC and sign in with your
   developer account.
2. Put the Jetson into recovery mode (hold the Recovery button while
   applying power, or follow your carrier board's manual) and connect it
   to the host PC over USB.
3. In SDK Manager, select **Jetson Orin Nano** as the target and JetPack
   **6.2** or newer, and flash to the attached NVMe SSD rather than the
   board's internal storage -- storage capacity is the tightest resource
   on this board, and repos, virtual environments, and logs belong on
   NVMe (see the NVMe setup section below).
4. After the flash completes and the board boots, verify the installed
   release from a terminal on the Jetson:

   ```sh
   cat /etc/nv_tegra_release
   ```

   The output should name an R36-series revision, which corresponds to
   JetPack 6.2 or newer.

## NVMe setup

`[UNVERIFIED]` Exact device names (`/dev/nvme0n1`, etc.) depend on the
specific SSD and carrier board; confirm with `lsblk` before running any
command that writes to a device.

If JetPack was not flashed directly to the NVMe drive, or a second drive
is present for extra storage, prepare it manually:

```sh
# Confirm the device name first -- getting this wrong can overwrite the
# wrong disk.
lsblk

sudo parted /dev/nvme0n1 --script mklabel gpt mkpart primary ext4 0% 100%
sudo mkfs.ext4 /dev/nvme0n1p1
sudo mkdir -p /mnt/nvme
sudo mount /dev/nvme0n1p1 /mnt/nvme

# Persist the mount across reboots.
uuid="$(sudo blkid -s UUID -o value /dev/nvme0n1p1)"
echo "UUID=$uuid  /mnt/nvme  ext4  defaults  0  2" | sudo tee -a /etc/fstab

# Relocate the projects directory so repos, virtual environments, and
# session logs live on NVMe instead of the space-constrained internal
# storage.
sudo mkdir -p /mnt/nvme/projects
sudo chown "$USER:$USER" /mnt/nvme/projects
mv "$HOME/projects" /mnt/nvme/projects 2>/dev/null || true
ln -s /mnt/nvme/projects "$HOME/projects"
```

## Power mode

Set the board to its highest-performance power profile before running
anything performance-sensitive (Kestrel's REPL, its test suite, or a
later interactive interface):

```sh
sudo nvpmodel -m 0     # MAXN: every core and clock domain unlocked
sudo jetson_clocks     # pin clocks to their MAXN ceiling
```

`nvpmodel -m 0` persists across reboots; `jetson_clocks` does not, and
must be re-run after every boot (add it to a login script or a systemd
unit if you want it applied automatically). Verify both at any time:

```sh
sudo nvpmodel -q        # should report "NV Power Mode: MAXN"
sudo jetson_clocks --show
```

## Python & uv

Kestrel is packaged and run entirely through `uv`; no separate
system-Python setup is required.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"    # or open a new shell

uv python install 3.12
uv python list                    # confirm 3.12 is present
```

## Kestrel install

```sh
git clone https://github.com/aruneem-bhowmick/kestrel.git
cd kestrel
uv sync --frozen
```

Set your model backend credentials as environment variables in your
shell profile (`~/.bashrc` or equivalent) -- never in a config file, and
never committed anywhere:

```sh
export OPENROUTER_API_KEY=sk-...redacted
export ZAI_API_KEY=sk-...redacted
```

Only one of the two is required; the flight check below will tell you if
the one your configuration expects is missing.

## Flight check

Before starting Kestrel itself, confirm the environment preconditions
this guide set up:

```sh
bash scripts/jetson-flightcheck.sh
```

Every line should read `PASS`; a `FAIL` line names exactly what to fix
and points back at the section above that covers it.

Then run Kestrel's own diagnostics:

```sh
uv run kestrel doctor
```

A healthy checkout with one credential set looks like this:

```text
OK    python-version  3.12
OK    config          ./kestrel.toml
OK    registry        2 models
OK    default-model   glm-5.2
OK    api-key         OPENROUTER_API_KEY
SKIP  endpoint        pass --live
SKIP  sandbox         sandboxed tool execution is not implemented
SKIP  ollama          the Ollama backend is not implemented
```

Finally, start the REPL and confirm a real completion streams back:

```sh
uv run kestrel
```

Type a message at the `kestrel>` prompt and watch the response stream in
token by token, followed by a cost line (`in:... out:... · $... turn ·
$... session`). Type `/quit` to exit.

## Ollama (deferred)

This board's GPU is also meant to serve a small local model (3B
parameters or fewer) through Ollama, for tasks like embeddings and cheap
classification that don't need a full remote model call. That
integration does not exist in this codebase yet; installing and
configuring Ollama is deferred to a later revision of this guide.
