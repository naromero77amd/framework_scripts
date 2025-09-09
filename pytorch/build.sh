#!/usr/bin/bash
# AOTRITONBASEDIR=/home/niromero/root/installed_aotriton
# mkdir -p ${AOTRITONBASEDIR}/0.7b-rocm6.2
# cd ${AOTRITONBASEDIR}/0.7b-rocm6.2
# wget https://github.com/ROCm/aotriton/releases/download/0.7b/aotriton-0.7b-manylinux_2_17_x86_64-rocm6.2-shared.tar.gz
# tar zvxf aotriton-0.7b-manylinux_2_17_x86_64-rocm6.2-shared.tar.gz
# export AOTRITON_INSTALLED_PREFIX=${AOTRITONBASEDIR}/0.7b-rocm6.2/aotriton

# next three lines needed when switching between branches
python3 setup.py clean # needed when switching branches
git submodule sync --recursive # synchronizes url in .gitmodules, do this recursively
git submodule foreach --recursive git reset --hard # cleanup in case there is a detached HEAD
git submodule update --init --recursive # clones missing submodules

# make clean # sometimes not needed
export MAX_JOBS=128
unset PYTORCH_ROCM_ARCH # much faster, but will only build to only the architecture found on the host
# export PYTORCH_ROCM_ARCH="gfx90a;gfx942"
# export PYTORCH_ROCM_ARCH=gfx90a
pip uninstall torch
python tools/amd_build/build_amd.py
# the three lines below are because aotriton build system is problematic 
# rm -rf torch/tmp
# rm -rf torch/src
# rm -rf /root/.triton
unset CPLUS_INCLUDE_PATH
# REL_WITH_DEB_INFO=1 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 python setup.py install 2>&1 | tee build.log
# Fedora
# export CMAKE_MODULE_PATH=/usr
# export ROCM_PATH=/usr
# REL_WITH_DEB_INFO=1 USE_KINETO=0 BUILD_TEST=0 USE_FBGEMM=0 USE_DISTRIBUTED=0 USE_CUDNN=0 python setup.py install 2>&1 | tee build.log # both roctx linked in, plus libroctracer
# REL_WITH_DEB_INFO=1 USE_KINETO=0 USE_FBGEMM=1 python setup.py install 2>&1 | tee build.log # both roctx linked in, but not libroctracer
# THE COMBINATION BELOW FOR ROCM 6.2.0 works
# export ROCM_PATH=/opt/rocm-6.2.0
# CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 LDFLAGS="-fuse-ld=gold" REL_WITH_DEB_INFO=1 USE_KINETO=0 USE_FBGEMM=0 python setup.py install 2>&1 | tee build.log # both roctx linked in, plus libroctracer
# CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ REL_WITH_DEB_INFO=1 USE_KINETO=0 USE_FBGEMM=0 python setup.py install 2>&1 | tee build.log # both roctx linked in, plus libroctracer
# LDFLAGS="-fuse-ld=gold" REL_WITH_DEB_INFO=1 USE_KINETO=0 USE_FBGEMM=0 python setup.py install 2>&1 | tee build.log # works
# USE_GOLD_LINKER=1 REL_WITH_DEB_INFO=1 USE_KINETO=0 USE_FBGEMM=0 python setup.py install 2>&1 | tee build.log # works
# REL_WITH_DEB_INFO=1 USE_NNPACK=0 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_KINETO=0 USE_FBGEMM=0 USE_MPI=0 USE_MAGMA=0 BUILD_TEST=0 python setup.py install 2>&1 | tee build.log # works
# do not do ROCM
USE_ROCM=1 REL_WITH_DEB_INFO=1 USE_NNPACK=0 USE_FLASH_ATTENTION=1 USE_MEM_EFF_ATTENTION=1 USE_KINETO=1 USE_FBGEMM=0 USE_MPI=0 USE_MAGMA=1 BUILD_TEST=0 python setup.py install 2>&1 | tee build.log # works
# USE_XPU=0 REL_WITH_DEB_INFO=1 USE_KINETO=0 USE_FBGEMM=0 python setup.py install 2>&1 | tee build.log # WAG doesn't work.
# USE_ROCM=1 REL_WITH_DEB_INFO=1 USE_NNPACK=0 USE_FLASH_ATTENTION=1 USE_MEM_EFF_ATTENTION=1 USE_KINETO=1 USE_FBGEMM=0 USE_MPI=0 USE_MAGMA=0 BUILD_TEST=0 python setup.py install