---
name: build-pytorch
description: Build PyTorch inside a Docker container using framework_scripts/pytorch/build.sh after docker-critical-tools has prepared the container. Use only when manually invoked with /build-pytorch.
disable-model-invocation: true
---

# Build PyTorch

## Prerequisite

Before using this skill, `/docker-critical-tools` must have been invoked successfully for the target Docker container.

That skill ensures the container has the required tools and that this repository is cloned and up to date:

```text
/home/niromero/docker_workspace/framework_scripts
```

If `/docker-critical-tools` has not been run for the target container, stop and run it first.

## Build Target

Default to building PyTorch for ROCm.

Only build for CUDA if the user explicitly asks for CUDA.

## Build Script

Use this script from the `framework_scripts` repository:

```text
/home/niromero/docker_workspace/framework_scripts/pytorch/build.sh
```

Before launching the build, verify the script exists and is executable:

```bash
test -x /home/niromero/docker_workspace/framework_scripts/pytorch/build.sh
```

If it exists but is not executable, run:

```bash
chmod +x /home/niromero/docker_workspace/framework_scripts/pytorch/build.sh
```

## Workflow

1. Identify the target Docker container.
2. Confirm `/docker-critical-tools` has already prepared the container. If not, run `/docker-critical-tools` first.
3. Use ROCm unless the user explicitly requested CUDA.
4. Start the build from the PyTorch scripts directory.
5. Write build output to `build.log`.
6. Report status every 5 minutes using the current contents of `build.log`.
7. When the build exits, report success or failure and summarize the final relevant lines from `build.log`.

## Build Commands

For ROCm, run the build from inside the container:

```bash
cd /home/niromero/docker_workspace/framework_scripts/pytorch
./build.sh 2>&1 | tee build.log
```

For CUDA, use the CUDA option supported by `build.sh`. If the script usage is unclear, inspect `build.sh` first and choose the CUDA path it defines. Do not guess unsupported flags.

## Docker Exec Form

For a running container, use this shape:

```bash
docker exec <container> bash -lc '
set -o pipefail
cd /home/niromero/docker_workspace/framework_scripts/pytorch
test -x ./build.sh || chmod +x ./build.sh
./build.sh 2>&1 | tee build.log
'
```

If building for CUDA, adjust only the `./build.sh` invocation according to the script's documented CUDA option.

## Status Reporting

While the build is running, report every 5 minutes based on `build.log`.

Each status update should include:

- Whether the build process is still running.
- The most recent meaningful build phase or command visible in `build.log`.
- The latest warnings or errors, if any.
- The last 20-40 lines of `build.log` summarized, not pasted wholesale unless the user asks.

Use `build.log` as the source of truth. Do not invent progress if the log is quiet.

Helpful command inside the PyTorch scripts directory:

```bash
tail -n 80 build.log
```

If `build.log` has not changed since the previous report, say that explicitly and include how long it has been quiet.

## Failure Handling

If the build fails:

1. Report the failing command or phase if visible.
2. Include the key error lines from `build.log`.
3. Do not automatically clean or restart the build unless the user asks.

If `build.sh` is missing, stop and ask the user to confirm that `/docker-critical-tools` ran successfully and that `framework_scripts` is present at the expected path.
