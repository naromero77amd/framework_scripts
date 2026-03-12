#!/usr/bin/bash

# Run this entire script in a tmux session because the tuning steps takes a while to run

# Use the new torch profile benchmarker for the most accurate timings
# export TORCHINDUCTOR_USE_TORCH_PROFILER_BENCHMARKER=1 ### need for kernels with short duration
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_paas ### optional to keep Inductor in a different cache
rm -rf ${TORCHINDUCTOR_CACHE_DIR}

kernels_dir="hrt_kernels"

# Assumes kernels have been extracted already
# paas-organize --inductor-cache=torchinductor_dream3_with_pruning --output-dir=${kernels_dir}

# Use 4 GPUs
export HIP_VISIBLE_DEVICES="0,1,2,3"

# required benchmarking
 paas-inductor --dir=${kernels_dir} --distributed --run-types=autotune

# standalone kernels
paas-make-standalone --mode minimal --launch-params-suffix=.autotune.launch_params ./${kernels_dir}/reduction
paas-make-standalone --mode tune --launch-params-suffix=.autotune.launch_params ./${kernels_dir}/reduction
paas-make-standalone --mode minimal --launch-params-suffix=.autotune.launch_params ./${kernels_dir}/pointwise
paas-make-standalone --mode tune --launch-params-suffix=.autotune.launch_params ./${kernels_dir}/pointwise

# exhaustive tuning with augement stages
cd ${kernels_dir}/reduction/
paas-simple-full 
paas-simple-full --augment-stages

cd ../..

cd ${kernels_dir}/pointwise/
paas-simple-full
paas-simple-full --augment-stages 
