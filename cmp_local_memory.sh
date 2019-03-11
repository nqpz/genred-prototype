#!/bin/sh
#
# We compare the different local-memory-using kernels.  For each of the three
# methods -- atomicAdd, atomicCAS, and atomicExch -- we compare the
# *_SHARED_CHUNK_COOP kernel with the *_SHARED_CHUNK_COOP_COL kernel on a
# variety of datasets.
#
# Kernel explanations:
#
#   - *_SHARED_CHUNK_COOP (*3): Uses local memory, chunking, and an adjustable
#     cooperation level (number of threads cooperating over a subhistogram).
#
#   - *_SHARED_CHUNK_COOP_COL (*4): The same, but associates subhistograms with
#     threads in a way that each thread in a warp accesses a unique
#     subhistogram, assuming the number of subhistograms in a block is a
#     multiple of the warp/wavefront size.  For example, if both the number of
#     subhistograms and the warp size is 32, we will get zero intra-warp
#     conflicts -- and even if the number of subhistograms is 16 or 8, we should
#     still at least see a reduction.
#
# For atomicExch we also compare with the kernels SHLOCK_SHARED_CHUNK_COOP_AEXCH
# and SHLOCK_SHARED_CHUNK_COOP_THREADID that use locks based on lock-step
# execution semantics, and only work under the assumption of number of
# subhistograms being aligned to the warp size.  Kernels:
#
#   - SHLOCK_SHARED_CHUNK_COOP_AEXCH: Keep using atomicExch, but do it in the
#     same piece of local memory as the histogram.
#
#   - SHLOCK_SHARED_CHUNK_COOP_THREADID: Use an ad-hoc locking mechanism using
#     thread ids in the same piece of local memory as the histogram.
#
# All of these kernels has a reduction pass at the end which we do not measure
# (because we do not implement an efficient reduce in this test framework).
#
# We would like to see if the *_COL kernels are never slower than the "base"
# kernels in edge cases, and always faster in common cases.
#
# Datasets:
#
#   - Random.
#
#   - All indices are the same.  All writes go to index 0.  This is adverserial
#     to all approaches, but the question is which one handles it the least
#     worst.
#
#   - No intra-warp conflicts even with the base (row) kernels, assuming a large
#     enough number of subhistograms (aligned with the warp size).  This is
#     designed to be adverserial towards the *_COL kernels, which are not
#     expected to cause a speedup here.
#
# We run each kernel 100 times just because it really doesn't take that long.

set -e # Exit on the first error.
make datcuda # Make sure the datasets exist.
make # Make sure the ./host executable exists.

new_page() {
    echo '\f' # page break
}

run() {
    kernel=$1
    coop_lvl=$2
    hist_size=$3
    input="$4"
    if ! [ "$input" ]; then
        input=$hist_size
    fi
    img_size="$5"
    if ! [ "$img_size" ]; then
        img_size=10000000
    fi
    ./host $kernel $coop_lvl $hist_size data/cuda/$input-$img_size.dat 100 0
}

main() {
    for hist_size in 16 64 256 1024 4096; do
        echo "  Dataset: random; histogram size: $hist_size"
        for id in 12 13 14 22 23 24 32 33 34 35 36; do
            echo "    $(run $id 0 $hist_size)"
        done
        echo
    done
    new_page

    for hist_size in 16 64 256 1024 4096; do
        echo "  Dataset: all indices the same; histogram size: $hist_size"
        for id in 12 13 14 22 23 24 32 33 34 35 36; do
            echo "    $(run $id 0 $hist_size zeros)"
        done
        echo
    done
    new_page

    hist_size=32
    echo "  Dataset: no conflicts for row-based cooperation; histogram size: $hist_size"
    for id in 12 13 14 22 23 24 32 33 34 35 36; do
        echo "    $(run $id 0 $hist_size no-conflicts-warp-$hist_size)"
    done
    echo
    hist_size=256
    echo "  Dataset: no conflicts for row-based cooperation; histogram size: $hist_size"
    for id in 12 13 14 22 23 24 32 33 34 35 36; do
        echo "    $(run $id 0 $hist_size no-conflicts-warp-$hist_size 9999872)"
    done
    new_page
}

main
