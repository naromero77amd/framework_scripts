---
name: docker-critical-tools
description: Install critical tools and sync framework_scripts inside a Docker container. Use only when manually invoked with /docker-critical-tools.
disable-model-invocation: true
---

# Docker Critical Tools

## Instructions

Only run this skill when the user manually invokes `/docker-critical-tools`.

When invoked, make sure these critical programs are installed inside the target Docker container:

- `gh`
- `ripgrep`
- `strace`

Also make sure this repository is cloned and up to date inside the container:

```text
https://github.com/naromero77amd/framework_scripts.git
```

The repository must live at:

```text
/home/niromero/docker_workspace/framework_scripts
```

## Workflow

1. Identify the target container.
2. Check whether the critical programs and `git` are already installed:

```bash
command -v gh >/dev/null && command -v rg >/dev/null && command -v strace >/dev/null && command -v git >/dev/null
```

3. If any are missing, install them with `apt` inside the container:

```bash
apt update
apt install -y gh ripgrep strace git
```

4. If the container does not run as root, use `sudo` when available:

```bash
sudo apt update
sudo apt install -y gh ripgrep strace git
```

5. Verify installation:

```bash
gh --version
rg --version
strace --version
git --version
```

6. Ensure the workspace directory exists:

```bash
mkdir -p /home/niromero/docker_workspace
```

7. Clone or update `framework_scripts`:

```bash
if [ ! -d /home/niromero/docker_workspace/framework_scripts/.git ]; then
  git clone https://github.com/naromero77amd/framework_scripts.git /home/niromero/docker_workspace/framework_scripts
else
  current_url="$(git -C /home/niromero/docker_workspace/framework_scripts remote get-url origin)"
  if [ "$current_url" != "https://github.com/naromero77amd/framework_scripts.git" ]; then
    echo "ERROR: framework_scripts exists but origin is $current_url"
    exit 2
  fi
  git -C /home/niromero/docker_workspace/framework_scripts fetch --prune origin
  git -C /home/niromero/docker_workspace/framework_scripts pull --ff-only
fi
```

Use `pull --ff-only` so local edits in the container are not overwritten. If it fails, report the failure and ask the user how to handle the local branch state.

## Docker Usage Notes

For a running container, execute the workflow through `docker exec`:

```bash
docker exec <container> sh -lc '
command -v gh >/dev/null && command -v rg >/dev/null && command -v strace >/dev/null && command -v git >/dev/null || { apt update && apt install -y gh ripgrep strace git; }
mkdir -p /home/niromero/docker_workspace
if [ ! -d /home/niromero/docker_workspace/framework_scripts/.git ]; then
  git clone https://github.com/naromero77amd/framework_scripts.git /home/niromero/docker_workspace/framework_scripts
else
  current_url="$(git -C /home/niromero/docker_workspace/framework_scripts remote get-url origin)"
  [ "$current_url" = "https://github.com/naromero77amd/framework_scripts.git" ] || { echo "ERROR: framework_scripts origin is $current_url"; exit 2; }
  git -C /home/niromero/docker_workspace/framework_scripts fetch --prune origin
  git -C /home/niromero/docker_workspace/framework_scripts pull --ff-only
fi
'
```

If `apt` is unavailable, stop and tell the user the container image does not appear to be Debian or Ubuntu based. Do not silently substitute another package manager unless the user asks for that.

If package installation fails because of permissions, explain that the container user lacks install permissions and ask whether to retry as root or rebuild the image with the packages included.
