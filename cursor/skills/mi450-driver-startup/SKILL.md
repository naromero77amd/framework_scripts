---
name: mi450-driver-startup
description: Applies MI450 post-driver startup settings and performance workarounds. Use when preparing an MI450 system after loading the AMDGPU driver, configuring scale-up performance, or applying the 50us kernel-time workaround.
---

# MI450 Driver Startup

Run this workflow after the MI450 driver starts. Execute the commands in order and stop on the first failure. These commands require passwordless or interactive `sudo` access and download scripts over HTTP for execution as root.

## 1. Load AMDGPU

Run:

```bash
sudo modprobe amdgpu gpu_recovery=0
```

Verify that the command exits successfully and `/sys/module/amdgpu/parameters/gpu_recovery` contains `0`.

## 2. Disable NUMA balancing

Run:

```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```

Verify that the command exits successfully and `/proc/sys/kernel/numa_balancing` contains `0`.

## 3. Apply MI450 performance settings

Enable Bash pipeline failure detection with `set -o pipefail`, then run each command separately in this order:

```bash
curl -sSL http://dcgpuval-storage.amd.com/users/muku/MI450x_ScaleUp_PerfScripts/MI450_disTxIdle_may13.py | sudo python3
curl -sSL http://dcgpuval-storage.amd.com/users/kstraube/set_tdc_limits_mi450.py | sudo python3
curl -sSL http://dcgpuval-storage.amd.com/users/jelui/mi45x_scripts/disable_gcea_link_mgr.py | sudo python3
curl -sSL http://dcgpuval-storage.amd.com/users/jelui/mi45x_scripts/set_cp_hpd_enable_offload_check.py | sudo python3
```

Require exit status `0` from every pipeline. Check the output for Python tracebacks, tool errors, failed writes, or failed readback validation.

## 4. Apply the 50us kernel-time workaround

With Bash pipeline failure detection enabled, run:

```bash
curl -fsSL http://dcgpuval-storage.amd.com/users/tifyeung/Perf/Perf_DisSdpDisc_MGCG.sh | sudo bash
```

Require exit status `0`. Confirm the output reports the expected MI450 register writes and successful readbacks, including disabled MGCG settings and `TxIdleHistDis = 1`.

## Report

Report success only after every command and verification passes. If anything fails, identify the failed step and include the relevant error output without continuing to later steps.
