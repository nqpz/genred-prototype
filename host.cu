#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include "kernels.cu.h"
#include "misc.cu.h"

/* x0: One histogram in global memory. One pixel per thread.
 * x1: One histogram in global memory. Chunking.
 * x2: Cooperation in global memory.   Chunking.
 * x3: Coop. in sh. and glob. memory.  Chunking.
 */
#define SEQUENTIAL 00

#define AADD_NOSHARED_NOCHUNK_FULLCOOP  10
#define AADD_NOSHARED_CHUNK_FULLCOOP    11
#define AADD_NOSHARED_CHUNK_COOP        12
#define AADD_SHARED_CHUNK_COOP          13
#define AADD_SHARED_CHUNK_COOP_WARP     14

#define ACAS_NOSHARED_NOCHUNK_FULLCOOP  20
#define ACAS_NOSHARED_CHUNK_FULLCOOP    21
#define ACAS_NOSHARED_CHUNK_COOP        22
#define ACAS_SHARED_CHUNK_COOP          23
#define ACAS_SHARED_CHUNK_COOP_WARP     24

#define AEXCH_NOSHARED_NOCHUNK_FULLCOOP 30
#define AEXCH_NOSHARED_CHUNK_FULLCOOP   31
#define AEXCH_NOSHARED_CHUNK_COOP       32
#define AEXCH_SHARED_CHUNK_COOP         33
#define AEXCH_SHARED_CHUNK_COOP_WARP    34
#define AEXCH_SHARED_CHUNK_COOP_SHLOCK_EXCH  35
#define AEXCH_SHARED_CHUNK_COOP_SHLOCK_ADHOC 36

// debugging
#define PRINT_INFO     1
#define PRINT_INVALIDS 1
#define PRINT_SEQ_TIME 0

// runtime
#define MICROS 1 // 0 will give runtime in millisecs.
#define PRINT_RUNTIME(time) (MICROS ? \
  printf("%lu\n", time) : printf("%.3f\n", time / 1000.0))

// misc
#define IN_T  int
#define OUT_T int
#define MY_OP Add<OUT_T>

const char* kernel_name(int kernel) {
  const char* name;
  switch(kernel) {
    /* Atomic add */
  case AADD_NOSHARED_NOCHUNK_FULLCOOP:
    name = "AADD_NOSHARED_NOCHUNK_FULLCOOP";
    break;
  case AADD_NOSHARED_CHUNK_FULLCOOP:
    name = "AADD_NOSHARED_CHUNK_FULLCOOP";
    break;
  case AADD_NOSHARED_CHUNK_COOP:
    name = "AADD_NOSHARED_CHUNK_COOP";
    break;
  case AADD_SHARED_CHUNK_COOP:
    name = "AADD_SHARED_CHUNK_COOP";
    break;
  case AADD_SHARED_CHUNK_COOP_WARP:
    name = "AADD_SHARED_CHUNK_COOP_WARP";
    break;

    /* Locking - CAS */
  case ACAS_NOSHARED_NOCHUNK_FULLCOOP:
    name = "ACAS_NOSHARED_NOCHUNK_FULLCOOP";
    break;
  case ACAS_NOSHARED_CHUNK_FULLCOOP:
    name = "ACAS_NOSHARED_CHUNK_FULLCOOP";
    break;
  case ACAS_NOSHARED_CHUNK_COOP:
    name = "ACAS_NOSHARED_CHUNK_COOP";
    break;
  case ACAS_SHARED_CHUNK_COOP:
    name = "ACAS_SHARED_CHUNK_COOP";
    break;
  case ACAS_SHARED_CHUNK_COOP_WARP:
    name = "ACAS_SHARED_CHUNK_COOP_WARP";
    break;

    /* Locking - Exch */
  case AEXCH_NOSHARED_NOCHUNK_FULLCOOP:
    name = "AEXCH_NOSHARED_NOCHUNK_FULLCOOP";
    break;
  case AEXCH_NOSHARED_CHUNK_FULLCOOP:
    name = "AEXCH_NOSHARED_CHUNK_FULLCOOP";
    break;
  case AEXCH_NOSHARED_CHUNK_COOP:
    name = "AEXCH_NOSHARED_CHUNK_COOP";
    break;
  case AEXCH_SHARED_CHUNK_COOP:
    name = "AEXCH_SHARED_CHUNK_COOP";
    break;
  case AEXCH_SHARED_CHUNK_COOP_WARP:
    name = "AEXCH_SHARED_CHUNK_COOP_WARP";
    break;
  case AEXCH_SHARED_CHUNK_COOP_SHLOCK_EXCH:
    name = "AEXCH_SHARED_CHUNK_COOP_SHLOCK_EXCH";
    break;
  case AEXCH_SHARED_CHUNK_COOP_SHLOCK_ADHOC:
    name = "AEXCH_SHARED_CHUNK_COOP_SHLOCK_ADHOC";
    break;
  case SEQUENTIAL:
    name = "SEQUENTIAL";
    break;
  default:
    name = "(unknown)";
    break;
  }
  return name;
}

int kernel_run(int kernel,
               IN_T *h_img,
               OUT_T *h_his,
               OUT_T *h_seq,
               int img_sz,
               int his_sz,
               int num_threads,
               int seq_chunk,
               int coop_lvl,
               int num_hists,
               struct timeval *t_start,
               struct timeval *t_end) {
  int res;
  switch(kernel) {
    /* Atomic add */
  case AADD_NOSHARED_NOCHUNK_FULLCOOP:
    res = aadd_noShared_noChunk_fullCoop<IN_T>
      (h_img, h_his, img_sz, his_sz, t_start, t_end,
       PRINT_INFO);
    break;
  case AADD_NOSHARED_CHUNK_FULLCOOP:
    res = aadd_noShared_chunk_fullCoop<IN_T>
      (h_img, h_his, img_sz, his_sz, num_threads,
       t_start, t_end, PRINT_INFO);
    break;
  case AADD_NOSHARED_CHUNK_COOP:
    res = aadd_noShared_chunk_coop<IN_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, coop_lvl, num_hists,
       t_start, t_end, PRINT_INFO);
    break;
  case AADD_SHARED_CHUNK_COOP:
    res = aadd_shared_chunk_coop<IN_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, coop_lvl, num_hists,
       t_start, t_end, PRINT_INFO);
    break;
  case AADD_SHARED_CHUNK_COOP_WARP:
    res = aadd_shared_chunk_coop_warp<IN_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, coop_lvl, num_hists,
       t_start, t_end, PRINT_INFO);
    break;

    /* Locking - CAS */
  case ACAS_NOSHARED_NOCHUNK_FULLCOOP:
    res = CAS_noShared_noChunk_fullCoop
      <MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz, t_start, t_end,
       PRINT_INFO);
    break;
  case ACAS_NOSHARED_CHUNK_FULLCOOP:
    res = CAS_noShared_chunk_fullCoop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz, num_threads,
       t_start, t_end, PRINT_INFO);
    break;
  case ACAS_NOSHARED_CHUNK_COOP:
    res = CAS_noShared_chunk_coop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, coop_lvl, num_hists,
       t_start, t_end, PRINT_INFO);
    break;
  case ACAS_SHARED_CHUNK_COOP:
    res = CAS_shared_chunk_coop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, coop_lvl, num_hists,
       t_start, t_end, PRINT_INFO);
    break;
  case ACAS_SHARED_CHUNK_COOP_WARP:
    res = CAS_shared_chunk_coop_warp<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, coop_lvl, num_hists,
       t_start, t_end, PRINT_INFO);
    break;

    /* Locking - Exch */
  case AEXCH_NOSHARED_NOCHUNK_FULLCOOP:
    res = exch_noShared_noChunk_fullCoop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz, t_start, t_end,
       PRINT_INFO);
    break;
  case AEXCH_NOSHARED_CHUNK_FULLCOOP:
    res = exch_noShared_chunk_fullCoop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz, num_threads, seq_chunk,
       t_start, t_end, PRINT_INFO);
    break;
  case AEXCH_NOSHARED_CHUNK_COOP:
    res = exch_noShared_chunk_coop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, coop_lvl, num_hists,
       t_start, t_end, PRINT_INFO);
    break;
  case AEXCH_SHARED_CHUNK_COOP:
    res = exch_shared_chunk_coop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, coop_lvl, num_hists,
       t_start, t_end, PRINT_INFO);
    break;
  case AEXCH_SHARED_CHUNK_COOP_WARP:
    res = exch_shared_chunk_coop_warp<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, coop_lvl, num_hists,
       t_start, t_end, PRINT_INFO);
    break;
  case AEXCH_SHARED_CHUNK_COOP_SHLOCK_EXCH:
    res = exch_shared_chunk_coop_shlock_exch<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, coop_lvl, num_hists,
       t_start, t_end, PRINT_INFO);
    break;
  case AEXCH_SHARED_CHUNK_COOP_SHLOCK_ADHOC:
    res = exch_shared_chunk_coop_shlock_adhoc<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, coop_lvl, num_hists,
       t_start, t_end, PRINT_INFO);
    break;
  case SEQUENTIAL:
    scatter_seq<MY_OP, IN_T, OUT_T>
      (h_img, h_seq, img_sz, his_sz, t_start, t_end);
    res = 0;
    break;
  default:
    res = 1;
    break;
  }
  return res;
}

int main(int argc, const char* argv[])
{
  /* validate and parse cmd-line arguments */
  int his_sz, kernel, coop_lvl_tmp, n_runs;
  if(validate_input(argc, argv,
                    &his_sz, &kernel, &coop_lvl_tmp, &n_runs) != 0) {
    return -1;
  }

  /* abort as soon as possible */
  unsigned int his_mem_sz = his_sz * sizeof(OUT_T);
  int restrict = (kernel == AADD_SHARED_CHUNK_COOP ||
                  kernel == AADD_SHARED_CHUNK_COOP_WARP ||
                  kernel == ACAS_SHARED_CHUNK_COOP ||
                  kernel == ACAS_SHARED_CHUNK_COOP_WARP ||
                  kernel == AEXCH_SHARED_CHUNK_COOP ||
                  kernel == AEXCH_SHARED_CHUNK_COOP_WARP);
  if(restrict && (his_mem_sz > SH_MEM_SZ)) {
    printf("Error: Histogram exceeds shared memory size\n");
    return -1;
  } else if(restrict && coop_lvl_tmp > BLOCK_SZ) {
    printf("Error: Cooperation level exceeds block size\n");
    return -1;
  }

  /* check that data file exists */
  if( access(argv[4], F_OK) == -1 ) {
    printf("Error: file '%s' does not exist\n", argv[4]);
    return 2;
  }

  /* get read handle */
  FILE *fp = fopen(argv[4], "r");
  if(fp == NULL) {
    printf("Error: Did not obtain read handle\n");
    return 3;
  }

  /* parse data file size (first number in file) */
  int img_sz;
  if(fscanf(fp, "%d", &img_sz) != 1) {
    printf("Error: Did not read data size\n");
    fclose(fp);
    return 4;
  }

  /* malloc host memory */
  IN_T  *h_img = (IN_T  *)malloc(img_sz * sizeof(IN_T));
  OUT_T *h_his = (OUT_T *)malloc(his_sz * sizeof(OUT_T));
  OUT_T *h_seq = (OUT_T *)malloc(his_sz * sizeof(OUT_T));

  /* parse data */
  int pixel;
  for(int i = 0; i < img_sz; i++) {
    if( fscanf(fp, "%d", &pixel) != 1) {
      printf("Error: Incorrect read\n");
      free(h_img); free(h_his); free(h_seq);
      fclose(fp);
      return 7;
    } else {
      h_img[i] = pixel;
    }
  }

  /* close file handle */
  fclose(fp);

  /* initialize result histograms with neutral element */
  initialize_histogram<MY_OP, OUT_T>(h_seq, his_sz);
  initialize_histogram<MY_OP, OUT_T>(h_his, his_sz);

  /* compute seq. chunk, coop. level and num. histos */
  // 1) N number of threads.
  int num_threads = NUM_THREADS(img_sz);

  // 2) varying coop. level
  int seq_chunk   = SEQ_CHUNK(img_sz, num_threads);
  num_threads = ceil(img_sz / (float)seq_chunk);

  int coop_lvl = 0;
  if(coop_lvl_tmp > num_threads) {
    coop_lvl = num_threads;
  } else if(coop_lvl_tmp == 0) {
    coop_lvl = COOP_LEVEL(his_sz, seq_chunk);
  } else {
    coop_lvl = coop_lvl_tmp;
  }
  int num_hists   = NUM_HISTOS(num_threads, coop_lvl);

  if(PRINT_INFO) {
    printf("== Heuristic formulas ==\n");
    if(kernel == AADD_NOSHARED_NOCHUNK_FULLCOOP ||
       kernel == ACAS_NOSHARED_NOCHUNK_FULLCOOP ||
       kernel == AEXCH_NOSHARED_NOCHUNK_FULLCOOP) {
      printf("Number of threads:    %d\n", img_sz);
      printf("Sequential chunk:     %d\n", 1);
      printf("Cooperation level:    %d\n", img_sz);
      printf("Number of histograms: %d\n", 1);
    } else if(kernel == AADD_NOSHARED_CHUNK_FULLCOOP ||
              kernel == ACAS_NOSHARED_CHUNK_FULLCOOP ||
              kernel == AEXCH_NOSHARED_CHUNK_FULLCOOP) {
      printf("Number of threads:    %d\n", num_threads);
      printf("Sequential chunk:     %d\n", seq_chunk);
      printf("Cooperation level:    %d\n", num_threads);
      printf("Number of histograms: %d\n", 1);
    } else {
      printf("Number of threads:    %d\n", num_threads);
      printf("Sequential chunk:     %d\n", seq_chunk);
      printf("Cooperation level:    %d\n", coop_lvl);
      printf("Number of histograms: %d\n", num_hists);
    }
    printf("====\n");
  }

  /** Kernel versions **/
  int res = 0;
  unsigned long int elapsed, elapsed_total, elapsed_avg;
  struct timeval t_start, t_end, t_diff;

  printf("Kernel: %s\n", kernel_name(kernel));
  elapsed_total = 0;
  for (int i = -1; i < n_runs; i++) {
    if (i < 0) {
      puts("Warmup run.");
      res = kernel_run(kernel, h_img, h_his, h_seq, img_sz, his_sz, num_threads, seq_chunk, coop_lvl, num_hists, &t_start, &t_end);
      if(res != 0) {
        free(h_img); free(h_his); free(h_seq);
        return res;
      }
    } else {
      printf("Run %d:\n", i);
      res = kernel_run(kernel, h_img, h_his, h_seq, img_sz, his_sz, num_threads, seq_chunk, coop_lvl, num_hists, &t_start, &t_end);

      if(res != 0) {
        free(h_img); free(h_his); free(h_seq);
        return res;
      }

      /* compute elapsed time for parallel version */
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;
      PRINT_RUNTIME(elapsed);
      printf("====\n");
      elapsed_total += elapsed;
    }
  }
  elapsed_avg = elapsed_total / n_runs;

  /* execute sequential scatter */
  unsigned long int seq_elapsed;
  struct timeval seq_start, seq_end, seq_diff;

  scatter_seq<MY_OP, IN_T, OUT_T>
    (h_img, h_seq, img_sz, his_sz, &seq_start, &seq_end);

  /* compute elapsed time for sequential version */
  timeval_subtract(&seq_diff, &seq_end, &seq_start);
  seq_elapsed = seq_diff.tv_sec * 1e6 + seq_diff.tv_usec;

  /* validate the last result */
  PRINT_RUNTIME(elapsed_avg);
  int valid = validate_array<OUT_T>(h_his, h_seq, his_sz);
  if(!valid) { printf("ERROR: Invalid!\n"); res = -1; }

  if(!valid && PRINT_INVALIDS) {
    print_invalid_indices<MY_OP, OUT_T>(h_his, h_seq, his_sz);
  }

  if(PRINT_SEQ_TIME) { PRINT_RUNTIME(seq_elapsed); }

  /* free host memory */
  free(h_img); free(h_his); free(h_seq);

  return 0;
}
