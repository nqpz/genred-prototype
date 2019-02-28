#ifndef MISC_H
#define MISC_H

#include <math.h>
#include <cassert>

//#define HWD       (61440)
#define HWD       (30720)
#define BLOCK_SZ  (1024)
#define SH_MEM_SZ (48 * 1024)
#define WARP_SZ   (32)

#define NUM_THREADS(n)  min(n, HWD)

#define SEQ_CHUNK(x, y)  ceil(x / (float)y)
#define COOP_LEVEL(x, y) ceil(x / (float)y)
#define NUM_HISTOS(x, y) ceil(x / (float)y)

#define BLOCK_X_DIM(x) (x < BLOCK_SZ ? x : BLOCK_SZ)
#define GRID_X_DIM(x)  ((x + (BLOCK_X_DIM(x) - 1)) / BLOCK_X_DIM(x))

/* Operators - should be associative and commutative */
template<class T>
class Add
{
 public:
  typedef T BaseType;
  static __device__ __host__ inline T identity() { return (T)0; }
  static __device__ __host__ inline T apply(const T t1, const T t2) { return t1 + t2; }
};

template<class T>
class Mul
{
 public:
  typedef T BaseType;
  static __device__ __host__ inline T identity() { return (T)1;    }
  static __device__ __host__ inline T apply(const T t1, const T t2) { return t1 * t2; }
};

/* Compute difference between two timeval structs. */
int timeval_subtract(struct timeval* result,
                     struct timeval* t2,
                     struct timeval* t1)
{
  unsigned int resolution=1000000;
  long int diff = (t2->tv_usec + resolution * t2->tv_sec) -
    (t1->tv_usec + resolution * t1->tv_sec);
  result->tv_sec = diff / resolution;
  result->tv_usec = diff % resolution;

  return (diff<0);
}

/* Validate input */
int validate_input(int argc, const char* argv[],
                   int *his_sz, int *kernel, int *coop_lvl, int *n_runs, int *print_info)
{
  /* check number of arguments */
  if(!(argc == 5 || argc == 6 || argc == 7)) {
    printf("Usage: "
           "%s <kernel> <coop. level> <histo. size> "
           "<filename> [<n runs>=10 [<print info=1>]]\n",
           argv[0]);
    return 1;
  }

  /* parse kernel type */
  if( sscanf(argv[1], "%i", kernel ) != 1) {
    printf("Error: Failed to parse kernel type\n");
  }

  /* parse cooporation level */
  if( sscanf(argv[2], "%i", coop_lvl ) != 1) {
    printf("Error: Failed to parse cooporation level\n");
  }

  /* parse histogram size */
  if( sscanf(argv[3], "%i", his_sz ) != 1) {
    printf("Error: Failed to parse histogram size\n");
  }

  if (argc == 6) {
    if( sscanf(argv[5], "%i", n_runs ) != 1) {
      printf("Error: Failed to parse number of runs\n");
    }
    assert(*n_runs > 0);
  } else {
    *n_runs = 10;
  }
  if (argc == 7) {
    if( sscanf(argv[6], "%i", print_info ) != 1) {
      printf("Error: Failed to parse whether to print info (specify 0 or 1)\n");
    }
    assert(*print_info == 0 || *print_info == 1);
  } else {
    *print_info = 1;
  }

  return 0;
}

/* Initialize histograms */
template<class OP, class T>
void
initialize_histogram(T *his, int his_sz)
{
  for(int i=0; i < his_sz; i++) {
    his[i] = OP::identity();
  }
}

/* Given two arrays, print indices where they differ. */
template<class OP, class T>
void
print_invalid_indices(T *par, T *seq, int sz)
{
  for(int i=0; i<sz; i++) {
    if(par[i] != seq[i]) {
      printf("idx: %d\npar: %d\nseq: %d\n\n", i, par[i], seq[i]);
    }
  }
}

/* Given an array, print values at all indices. */
template<class T>
void
print_array(T *data, int sz)
{
  for(int i=0; i < sz; i++) {
    printf("idx: %d: %d\n", i, data[i]);
  }
}

/* Given two arrays, check for equality on all indices. */
template<class T>
int
validate_array(T *par, T *seq, int sz)
{
  for(int i=0; i < sz; i++) {
    if(par[i] != seq[i]) { return 0; }
  }

  return 1;
}

#endif // MISC_H
