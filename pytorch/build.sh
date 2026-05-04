#!/usr/bin/bash
set -Eeuo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_pytorch_checkout() {
  [[ -f setup.py && -d .git && -d tools/amd_build ]] || die "Run this script from the PyTorch repository root."
}

print_dirty_submodules() {
  git submodule foreach --quiet --recursive '
    if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
      echo
      echo "Dirty submodule: $displaypath"
      git status --short
    fi
  '
}

dirty_submodules() {
  git submodule foreach --quiet --recursive '
    if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
      echo "$displaypath"
    fi
  '
}

misaligned_submodules() {
  git submodule status --recursive | while IFS= read -r line; do
    case "$line" in
      [-+U]*) echo "$line" ;;
    esac
  done
}

print_submodule_state() {
  local misaligned dirty
  misaligned="$(misaligned_submodules)"
  dirty="$(dirty_submodules)"

  if [[ -n "$misaligned" ]]; then
    echo
    echo "Submodules not at the commits recorded by PyTorch:"
    echo "$misaligned"
  fi

  if [[ -n "$dirty" ]]; then
    print_dirty_submodules
  fi
}

submodules_are_clean() {
  [[ -z "$(misaligned_submodules)" && -z "$(dirty_submodules)" ]]
}

reset_submodules() {
  cat <<'EOF'

PYTORCH_RESET_SUBMODULES=1 is set. Resetting and cleaning submodules so they
match the commits recorded by the PyTorch superproject.

EOF
  git submodule foreach --recursive 'git reset --hard && git clean -fdx'
}

sync_submodules() {
  git submodule sync --recursive

  if [[ "${PYTORCH_RESET_SUBMODULES:-0}" == "1" ]]; then
    reset_submodules
    git submodule update --init --recursive
  elif ! git submodule update --init --recursive; then
    echo
    echo "Submodule checkout failed. This usually means one or more submodules"
    echo "have local changes or are checked out at commits that conflict with"
    echo "the commit recorded by the PyTorch superproject."
    print_submodule_state
    cat >&2 <<'EOF'

Refusing to continue because building with misaligned submodules can fail later
with confusing compiler errors.

To preserve local submodule work, inspect the dirty submodules above and stash,
commit, or clean them manually.

For a from-scratch build where it is OK to discard local submodule changes, rerun:
  PYTORCH_RESET_SUBMODULES=1 bash /home/niromero/docker_workspace/framework_scripts/pytorch/build.sh

EOF
    exit 1
  fi

  if ! submodules_are_clean; then
    print_submodule_state
    cat >&2 <<'EOF'

Refusing to continue because one or more submodules are dirty or still not at
the commits recorded by the PyTorch superproject.

To preserve local submodule work, inspect the dirty submodules above and stash,
commit, or clean them manually.

For a from-scratch build where it is OK to discard local submodule changes, rerun:
  PYTORCH_RESET_SUBMODULES=1 bash /home/niromero/docker_workspace/framework_scripts/pytorch/build.sh

EOF
    exit 1
  fi
}

require_pytorch_checkout
# Optional if you have a pre-built AOTriton, otherwise, default behavior is to download it.
# Build AOTriton is slow, so leave section below alone.
# AOTRITONBASEDIR=/home/niromero/root/installed_aotriton
# mkdir -p ${AOTRITONBASEDIR}/0.7b-rocm6.2
# cd ${AOTRITONBASEDIR}/0.7b-rocm6.2
# wget https://github.com/ROCm/aotriton/releases/download/0.7b/aotriton-0.7b-manylinux_2_17_x86_64-rocm6.2-shared.tar.gz
# tar zvxf aotriton-0.7b-manylinux_2_17_x86_64-rocm6.2-shared.tar.gz
# export AOTRITON_INSTALLED_PREFIX=${AOTRITONBASEDIR}/0.7b-rocm6.2/aotriton

# Keep submodules aligned before uninstalling torch or starting an expensive build.
# Set PYTORCH_RESET_SUBMODULES=1 for a from-scratch build that may discard local
# changes inside submodules.
sync_submodules

# Needed when switching branches, as generated files and build metadata can
# change in significant ways.
python3 setup.py clean # needed when switching branches
# git clean . -dfx # some times also needed due to leftoever files due to submodules that have been removed

export MAX_JOBS=128 # use as many CPU cores as possible to build PyTorch
unset PYTORCH_ROCM_ARCH # much faster, only build the architecture found on the host
# export PYTORCH_ROCM_ARCH="gfx90a;gfx942"
# export PYTORCH_ROCM_ARCH=gfx90a
pip uninstall -y torch # otherwise, the final PyTorch install could fail
python tools/amd_build/build_amd.py # hipification

# Legacy, only needed if building AOTriton from scratch.
# the three lines below are because aotriton build system is problematic 
# rm -rf torch/tmp
# rm -rf torch/src
# rm -rf /root/.triton

unset CPLUS_INCLUDE_PATH # needed on some internal AMD docker images.

# Build system assumes ROCM_PATH=/opt/rocm, set explictly if another version of ROCm is going to be used.
# export ROCM_PATH=/opt/rocm-6.2.0

# Build a subset of PyTorch features, balance of build speed with functionality.
# don't build MPI support, FBGEMM, or compiled-tests
USE_ROCM=1 REL_WITH_DEB_INFO=1 USE_FBGEMM=0 USE_MPI=0 BUILD_TEST=0 python setup.py install 2>&1 | tee build.log # works
