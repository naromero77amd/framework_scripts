#!/usr/bin/bash 
# NOTE: Always confirm that Triton is properly installed
#       by doing: pip show triton

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# Set TRITON_PATCHED_LLVM=1 to build LLVM from source and link Triton against
# it. Default 0 downloads the prebuilt LLVM pinned in cmake/llvm-info.json.
TRITON_PATCHED_LLVM="${TRITON_PATCHED_LLVM:-0}"
case "$TRITON_PATCHED_LLVM" in
  0|1) ;;
  *) die "TRITON_PATCHED_LLVM must be 0 or 1." ;;
esac

if [[ "$TRITON_PATCHED_LLVM" == "1" ]]; then
  # gfx1250 codegen fixes from triton-lang/llvm-project#4.
  export LLVM_COMMIT_HASH="${LLVM_COMMIT_HASH:-546e93aa7ff1c3f0bac60278c9d84e5b4d339a02}"
  export LLVM_PROJECT_URL="${LLVM_PROJECT_URL:-https://github.com/triton-lang/llvm-project}"
  export LLVM_PROJECT_PATH="${LLVM_PROJECT_PATH:-$PWD/llvm-project-gfx1250}"
  export LLVM_BUILD_PATH="${LLVM_BUILD_PATH:-$LLVM_PROJECT_PATH/build}"
  # LLVM_SYSPATH is the only LLVM variable setup.py forwards to CMake, and
  # setting it also suppresses the prebuilt LLVM download.
  export LLVM_SYSPATH="$LLVM_BUILD_PATH"
fi

pip uninstall -y pytorch-triton-rocm
pip uninstall -y triton
unset CMAKE_PREFIX_PATH
python setup.py clean
pip install -r python/requirements.txt

if [[ "$TRITON_PATCHED_LLVM" == "1" ]]; then
  # Runs after the requirements install because it needs cmake and ninja. The
  # first run clones and builds LLVM; later runs rebuild incrementally.
  scripts/build-llvm-project.sh || die "LLVM build failed, so Triton was not built."
fi

pip install . # triton 3.4 and later
# pip install python # triton 3.3
