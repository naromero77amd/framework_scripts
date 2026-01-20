#!/usr/bin/bash

# Run this entire script in a tmux session because the tuning steps takes a while to run

# Use the new torch profile benchmarker for the most accurate timings
TORCHINDUCTOR_USE_TORCH_PROFILER_BENCHMARKER=1

kernels_dir="kernels-dream3-with-pruning"

# Assumes kernels have been extracted already
# paas-organize --inductor-cache=torchinductor_dream3_with_pruning --output-dir=${kernels_dir}

# Use 4 GPUs
export HIP_VISIBLE_DEVICES="3,4,5,6"

# required benchmarking
 paas-inductor --dir=${kernels_dir} --distributed --run-types=autotune

# standalone kernels
paas-make-standalone --mode minimal --launch-params-suffix=.autotune.launch_params ./${kernels_dir}/persistent_reduction
paas-make-standalone --mode tune --launch-params-suffix=.autotune.launch_params ./${kernels_dir}/persistent_reduction
paas-make-standalone --mode minimal --launch-params-suffix=.autotune.launch_params ./${kernels_dir}/pointwise
paas-make-standalone --mode tune --launch-params-suffix=.autotune.launch_params ./${kernls_dir}/pointwise

# exhaustive tuning with augement stages
cd ${kernels_dir}/persistent_reduction/
paas-simple-full 
paas-simple-full --augment-stages

cd ../..

cd ${kernels_dir}/pointwise/
paas-simple-full
paas-simple-full --augment-stages 
