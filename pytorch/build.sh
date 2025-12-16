#!/usr/bin/bash
# Optional if you have a pre-built AOTriton, otherwise, default behavior is too download it.
# Build AOTriton is slow, so leave section below alone.
# AOTRITONBASEDIR=/home/niromero/root/installed_aotriton
# mkdir -p ${AOTRITONBASEDIR}/0.7b-rocm6.2
# cd ${AOTRITONBASEDIR}/0.7b-rocm6.2
# wget https://github.com/ROCm/aotriton/releases/download/0.7b/aotriton-0.7b-manylinux_2_17_x86_64-rocm6.2-shared.tar.gz
# tar zvxf aotriton-0.7b-manylinux_2_17_x86_64-rocm6.2-shared.tar.gz
# export AOTRITON_INSTALLED_PREFIX=${AOTRITONBASEDIR}/0.7b-rocm6.2/aotriton

# next two lines needed when switching between branches as many of the underlying submodules could have changed
# in significant ways.
python3 setup.py clean # needed when switching branches
# git clean . -dfx # some times also needed due to leftoever files due to submodules or hipification
git submodule sync --recursive # synchronizes url in .gitmodules, do this recursively
git submodule foreach --recursive git reset --hard # cleanup in case there is a detached HEAD
git submodule update --init --recursive # clones missing submodules

export MAX_JOBS=128 # use as many CPU cores as possible to build PyTorch
unset PYTORCH_ROCM_ARCH # much faster, only build the architecture found on the host
# export PYTORCH_ROCM_ARCH="gfx90a;gfx942"
# export PYTORCH_ROCM_ARCH=gfx90a
pip uninstall torch # otherwise, they finally PyTorch install could fail
python tools/amd_build/build_amd.py # hipification

# Legacy, only needed if building AOTriton from scratch.
# the three lines below are because aotriton build system is problematic 
# rm -rf torch/tmp
# rm -rf torch/src
# rm -rf /root/.triton

unset CPLUS_INCLUDE_PATH # needed on some internal AMD docker images.

# Build system assumes ROCM_PATH=/opt/rocm, set explictly if another version of ROCm is going to be used.
# export ROCM_PATH=/opt/rocm-6.2.0

# Build a balance subset of PyTorch features
USE_ROCM=1 REL_WITH_DEB_INFO=1 USE_NNPACK=0 USE_FLASH_ATTENTION=1 USE_MEM_EFF_ATTENTION=1 USE_KINETO=1 USE_FBGEMM=0 USE_MPI=0 USE_MAGMA=1 BUILD_TEST=0 python setup.py install 2>&1 | tee build.log # works
