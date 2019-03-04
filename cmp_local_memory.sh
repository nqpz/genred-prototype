#!/bin/sh
#
# We compare the different local-memory-using kernels.  For each of the three
# methods -- atomicAdd, atomicCAS, and atomicExch -- we compare the
# *_SHARED_CHUNK_COOP kernel with the *_SHARED_CHUNK_COOP_WARP kernel on a
# variety of datasets.
#
# Kernel explanations:
#
#   - *_SHARED_CHUNK_COOP (*3): Uses local memory, chunking, and an adjustable
#     cooperation level (number of threads cooperating over a subhistogram).
#
#   - *_SHARED_CHUNK_COOP_WARP (*4): The same, but associates subhistograms with
#     threads in a way that each thread in a warp accesses a unique
#     subhistogram, assuming the number of subhistograms in a block is a
#     multiple of the warp/wavefront size.  For example, if both the number of
#     subhistograms and the warp size is 32, we will get zero intra-warp
#     conflicts -- and even if the number of subhistograms is 16 or 8, we should
#     still at least see a reduction.
#
# All of these kernels has a reduction pass at the end which we do not measure
# (because we do not implement an efficient reduce in this test framework).
#
# We would like to see if the *_WARP kernels are never slower than the "base"
# kernels in edge cases, and always faster in common cases.  When running this
# script, for each section, check whether the second line has a lower number
# (the runtime) than the first line.
#
# Datasets:
#
#   - Random.
#
#   - No intra-warp conflicts even with the base kernels.  This is designed to
#     be adverserial towards the *_WARP kernels, which are not expected to cause
#     a speedup here.
#
#   - All indices are the same.  All writes go to index 0.  This is adverserial
#     to all approaches, but the question is which one handles it the least
#     worst.
#
#   - Some intra-warp conflicts.  Instead of 32 subhistograms per block we have
#     16 (4 with atomicExch due to lock storage overhead), so assuming a warp
#     size of 32 this can cause some intra-warp conflicts, but we assume that
#     the *_WARP kernels still perform better than the base kernels.
#
#   - More intra-warp conflicts.  Fewer subhistograms per block.  The speedup,
#   - if any, should be lower.
#
# In the end we also compare atomicExch with the kernels
# AEXCH_SHARED_CHUNK_COOP_SHLOCK_EXCH and AEXCH_SHARED_CHUNK_COOP_SHLOCK_ADHOC
# that use lock-step execution locks.  These only work under the assumption of
# lock-step execution *and* when the histogram size is at most the size of a
# warp.  Kernels:
#
#   - AEXCH_SHARED_CHUNK_COOP_SHLOCK_EXCH: Keep using atomicExch, but do it in
#     the same piece of local memory as the histogram.
#
#   - AEXCH_SHARED_CHUNK_COOP_SHLOCK_ADHOC: Use an ad-hoc locking mechanism
#     using thread ids in the same piece of local memory as the histogram.
#
# For each section, the first line is the base kernel (AEXCH_SHARED_CHUNK_COOP),
# the second line is the _WARP kernel, the third line is
# AEXCH_SHARED_CHUNK_COOP_SHLOCK_EXCH, and the fourth line is
# AEXCH_SHARED_CHUNK_COOP_SHLOCK_ADHOC.
#
# We run each kernel 100 times just because it really doesn't take that long.

set -e

make

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
    echo atomicAdd, random
    run 13 32 256
    run 14 32 256
    echo atomicAdd, no intra-warp conflicts
    run 13 32 256 no-conflicts-warp-256 9999872
    run 14 32 256 no-conflicts-warp-256 9999872
    echo atomicAdd, all indices the same
    run 13 32 256 zeros
    run 14 32 256 zeros
    echo atomicAdd, some intra-warp conflicts
    run 13 64 1024
    run 14 64 1024
    echo atomicAdd, more intra-warp conflicts
    run 13 320 4096
    run 14 320 4096
    echo

    echo atomicCAS, random
    run 23 32 256
    run 24 32 256
    echo atomicCAS, no intra-warp conflicts
    run 23 32 256 no-conflicts-warp-256 9999872
    run 24 32 256 no-conflicts-warp-256 9999872
    echo atomicCAS, all indices the same
    run 23 32 256 zeros
    run 24 32 256 zeros
    echo atomicCAS, some intra-warp conflicts
    run 23 64 1024
    run 24 64 1024
    echo atomicCAS, more intra-warp conflicts
    run 23 320 4096
    run 24 320 4096
    echo

    echo atomicExch, random
    run 33 32 256
    run 34 32 256
    echo atomicExch, no intra-warp conflicts
    run 33 32 256 no-conflicts-warp-256 9999872
    run 34 32 256 no-conflicts-warp-256 9999872
    echo atomicExch, all indices the same
    run 33 32 256 zeros
    run 34 32 256 zeros
    echo atomicExch, some intra-warp conflicts
    run 33 256 1024
    run 34 256 1024
    echo atomicExch, more intra-warp conflicts
    run 33 1024 4096
    run 34 1024 4096
    echo

    echo lock-step execution locks, random
    run 33 32 32
    run 34 32 32
    run 35 32 32
    run 36 32 32
    echo lock-step execution locks, no intra-warp conflicts
    run 33 32 32 no-conflicts-warp-32
    run 34 32 32 no-conflicts-warp-32
    run 35 32 32 no-conflicts-warp-32
    run 36 32 32 no-conflicts-warp-32
    echo lock-step execution locks, all indices the same
    run 33 32 32 zeros
    run 34 32 32 zeros
    run 35 32 32 zeros
    run 36 32 32 zeros
    echo
}

main
