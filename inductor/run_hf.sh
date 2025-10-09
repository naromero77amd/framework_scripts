#!/usr/bin/bash
set -u # exit on undefined variable
run_name="hf_baseline"
export CUDA_VISIBLE_DEVICES=7
export AMDGCN_USE_BUFFER_OPS=1
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1
export TORCHINDUCTOR_BENCHMARK_KERNEL=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/${run_name}
pytorchdir="/home/niromero/docker_workspace/pytorch"
outdir="/home/niromero/docker_workspace/meta_pr/${run_name}"

# Setup
timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p ${TORCHINDUCTOR_CACHE_DIR}
mkdir -p $outdir

date

# Clear Inductor Cache
rm -rf $TORCHINDUCTOR_CACHE_DIR

cd ${pytorchdir}
python benchmarks/dynamo/runner.py --dtypes amp --suites huggingface --training --compilers inductor_no_cudagraphs --no-gh-comment --output-dir $outdir

date

# Store inductor cache
tar -zcvf $outdir/inductor-cache-${run_name}-${timestamp}.tar.gz ${TORCHINDUCTOR_CACHE_DIR}
