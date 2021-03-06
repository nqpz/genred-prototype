# Prototyping of `reduce_by_index`

This repository is based on the excellent prototyping done by Sune
Hellfritzsch in his MSc thesis on implementing a generalised reduction
operator in the [Futhark](https://github.com/diku-dk/futhark/)
compiler.  We extend this.

Main files: `host.cu`, `kernels.cu.h`, and `misc.cu.h`.

Besides the CUDA prototypes this also contains a setup for replicating
performance tests.


## System requirements

You will need:

 * A working CUDA installation.

 * A working Futhark installation, i.e., the `futhark` executable must
   be in your `PATH`, and you need a working OpenCL installation.  See
   the installation instructions
   [here](https://futhark.readthedocs.io/en/latest/installation.html).

 * A working Python 2.7 installation (other versions might work too),
   with `Numpy` and `Matplotlib` installed.


## Compile and run binaries only

### CUDA

Make sure you are in the root Git directory (where this file
is located) and type `make host`. This will create an
executable named `host`. If you execute it without arguments
it will show which arguments it takes:

```
# ./host
Usage: ./host <kernel> <coop. level> <histo. size> <filename> [<n runs>=10 [<print info=1>]]
```

That is, it takes:

1. `<kernel>`: A kernel identifier. It can be one of the
following:

    . `[10|20|30]`: One histogram in global memory and one pixel per
    thread. That is, full cooperation and no chunking.

    . `[11|21|31]`: One histogram in global memory and chunking.

    . `[12|22|32]`: Multiple histograms in global memory and chunking.

    . `[13|23|33]`: Multiple histograms in both shared on global
    memory, and chunking.

    . `[14|24|34]`: Multiple histograms in both shared on global
    memory, and chunking, and warp optimizations.

    . `[35|36]`: Alternatives to `33`.  Uses the same shared memory
    for both histograms and locking.  `35` still uses `atomicExch`,
    while `36` tries to do its own thing.

    where versions prefixed by `1` are implemented using
    `atomicAdd`, versions prefixed by `2` are implemented by
    using `atomicCAS`, and versions prefixed by `3` are
    implemented by using `atomicExch` (except for 36). All
    methods implement a simple addition operation.

2. `<coop. level>`: How many threads should cooperate on one
histogram. If this is set to zero, the program will try and
compute a cooperation level for you. (We aim at computing
the optimal one.)

3. `<histo. size>`: Number of bins per histogram.

4. `<filename>`: The (relative or absolute) path, including
the filename, to the input file that you want to compute a
histogram over. See [Generating data](#generating-data) for
the details of how to produce data files with the correct layout.

Note that the runtimes reported are without the final reduction
phase, where intermediate histograms are combined.

### Futhark

`make reduce` will compile `reduce.fut` using the
`futhark opencl` compiler.


## Running the full performance test setup

The setup requires that the `data/cuda` and `data/futhark`
directories are present, and that the latter is populated
manually with data files (see [Futhark
data](#futhark-data)). Furthermore, a directory called
`runtimes/` should also be created. Now you are ready to run
`make plot` which will then take care of creating the CUDA
data files (but not the Futhark data files!), compiling the
sources, running the performance tests, and creating graphs
presenting the data. The intermediate json data files
containing the runtime information is placed in the
`runtime/` directory.


## Generating data

Just run `make plot` (or `make dat` if you only care about the data).
Read on for the details.


### CUDA data

If you want to test the CUDA prototypes on a single data set
at a time, you should create the data first by using
`generate_image.py`. It will generate a file containing
`<image size>` integers ranging from `0` to `<upper limit>`
both inclusive.

If you execute it without arguments it will show which
arguments it takes:

```
# python generate_image.py
Usage: python generate_image.py <upper limit> <image size>
```

That is, it takes:

 1. `<upper limit>`: The inclusive upper bound. This should
 be equal to the histogram size that you want to test, such
 that you get uniformly distributed data over the histogram
 size.

 2. `<image size>`: The number of integers to create.

This will create the data files in the `data/cuda`
directory.


### Futhark data

Run `./generate_fut_dat.sh` and futhark data files will be
generated in the `data/futhark` directory. Values are
hardcoded in the script and if they are changed, values in
other files should be changed manually as well.


### Sune's comments

Note, that the runtimes reported by the CUDA program are
without the final reduction phase, where intermediate
histograms are combined. This is due to the facts, that
writing an effective reduction in CUDA is non-trivial, that
it is not the scope of my thesis, and that the reduction
will by generated by the Futhark compiler anyway.

Beware that the setup is hardcoded for a specific range of
histogram sizes and cooperation levels. This is partly due
to the way the `futhark bench` and `futhark dataset` works,
and the problem structure.

The setup will run each CUDA kernel `5` times for each
combination of cooperation level and histogram size. These
values are set in `Makefile` but if they are changed, other
files must also be changed manually (so be careful).
Runtimes for the reduction phase will be computed by by
running `futhark bench` on `reduce.fut`. Each input file
will be run `5` times also.
