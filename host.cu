#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "kernels.cu.h"
#include "misc.cu.h"

#define SEQUENTIAL 00

#define AADD_GLOBAL_NOCHUNK_FULLCOOP      10
#define AADD_GLOBAL_CHUNK_FULLCOOP        11
#define AADD_GLOBAL_CHUNK_COOP            12
#define AADD_SHARED_CHUNK_COOP            13
#define AADD_SHARED_CHUNK_COOP_COL        14

#define ACAS_GLOBAL_NOCHUNK_FULLCOOP      20
#define ACAS_GLOBAL_CHUNK_FULLCOOP        21
#define ACAS_GLOBAL_CHUNK_COOP            22
#define ACAS_SHARED_CHUNK_COOP            23
#define ACAS_SHARED_CHUNK_COOP_COL        24

#define AEXCH_GLOBAL_NOCHUNK_FULLCOOP     30
#define AEXCH_GLOBAL_CHUNK_FULLCOOP       31
#define AEXCH_GLOBAL_CHUNK_COOP           32
#define AEXCH_SHARED_CHUNK_COOP           33
#define AEXCH_SHARED_CHUNK_COOP_COL       34
#define SHLOCK_SHARED_CHUNK_COOP_AEXCH    35
#define SHLOCK_SHARED_CHUNK_COOP_THREADID 36


// runtime
#define MICROS 1 // 0 will give runtime in millisecs.
#define PRINT_RUNTIME(time) (MICROS ? \
  printf("%lu", time) : printf("%.3f", time / 1000.0))

// misc
#define IN_T  int
#define OUT_T int
#define MY_OP Add<OUT_T>

const char* kernel_name(int kernel) {
  const char* name;
  switch(kernel) {
    /* Atomic add */
  case AADD_GLOBAL_NOCHUNK_FULLCOOP:
    name = "AADD_GLOBAL_NOCHUNK_FULLCOOP";
    break;
  case AADD_GLOBAL_CHUNK_FULLCOOP:
    name = "AADD_GLOBAL_CHUNK_FULLCOOP";
    break;
  case AADD_GLOBAL_CHUNK_COOP:
    name = "AADD_GLOBAL_CHUNK_COOP";
    break;
  case AADD_SHARED_CHUNK_COOP:
    name = "AADD_SHARED_CHUNK_COOP";
    break;
  case AADD_SHARED_CHUNK_COOP_COL:
    name = "AADD_SHARED_CHUNK_COOP_COL";
    break;

    /* Locking - CAS */
  case ACAS_GLOBAL_NOCHUNK_FULLCOOP:
    name = "ACAS_GLOBAL_NOCHUNK_FULLCOOP";
    break;
  case ACAS_GLOBAL_CHUNK_FULLCOOP:
    name = "ACAS_GLOBAL_CHUNK_FULLCOOP";
    break;
  case ACAS_GLOBAL_CHUNK_COOP:
    name = "ACAS_GLOBAL_CHUNK_COOP";
    break;
  case ACAS_SHARED_CHUNK_COOP:
    name = "ACAS_SHARED_CHUNK_COOP";
    break;
  case ACAS_SHARED_CHUNK_COOP_COL:
    name = "ACAS_SHARED_CHUNK_COOP_COL";
    break;

    /* Locking - Exch */
  case AEXCH_GLOBAL_NOCHUNK_FULLCOOP:
    name = "AEXCH_GLOBAL_NOCHUNK_FULLCOOP";
    break;
  case AEXCH_GLOBAL_CHUNK_FULLCOOP:
    name = "AEXCH_GLOBAL_CHUNK_FULLCOOP";
    break;
  case AEXCH_GLOBAL_CHUNK_COOP:
    name = "AEXCH_GLOBAL_CHUNK_COOP";
    break;
  case AEXCH_SHARED_CHUNK_COOP:
    name = "AEXCH_SHARED_CHUNK_COOP";
    break;
  case AEXCH_SHARED_CHUNK_COOP_COL:
    name = "AEXCH_SHARED_CHUNK_COOP_COL";
    break;
  case SHLOCK_SHARED_CHUNK_COOP_AEXCH:
    name = "SHLOCK_SHARED_CHUNK_COOP_AEXCH";
    break;
  case SHLOCK_SHARED_CHUNK_COOP_THREADID:
    name = "SHLOCK_SHARED_CHUNK_COOP_THREADID";
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
               int img_sz,
               int his_sz,
               int num_threads,
               int seq_chunk,
               int coop_lvl,
               int num_hists,
               int64_t *t_start,
               int64_t *t_end,
               int print_info) {
  int res;
  switch(kernel) {
    /* Atomic add */
  case AADD_GLOBAL_NOCHUNK_FULLCOOP:
    res = aadd_noShared_noChunk_fullCoop<IN_T>
      (h_img, h_his, img_sz, his_sz, t_start, t_end,
       print_info);
    break;
  case AADD_GLOBAL_CHUNK_FULLCOOP:
    res = aadd_noShared_chunk_fullCoop<IN_T>
      (h_img, h_his, img_sz, his_sz, num_threads,
       t_start, t_end, print_info);
    break;
  case AADD_GLOBAL_CHUNK_COOP:
    res = aadd_noShared_chunk_coop<IN_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, coop_lvl, num_hists,
       t_start, t_end, print_info);
    break;
  case AADD_SHARED_CHUNK_COOP:
    res = aadd_shared_chunk_coop<IN_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, coop_lvl, num_hists,
       t_start, t_end, print_info);
    break;
  case AADD_SHARED_CHUNK_COOP_COL:
    res = aadd_shared_chunk_coop_col<IN_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, coop_lvl, num_hists,
       t_start, t_end, print_info);
    break;

    /* Locking - CAS */
  case ACAS_GLOBAL_NOCHUNK_FULLCOOP:
    res = CAS_noShared_noChunk_fullCoop
      <MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz, t_start, t_end,
       print_info);
    break;
  case ACAS_GLOBAL_CHUNK_FULLCOOP:
    res = CAS_noShared_chunk_fullCoop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz, num_threads,
       t_start, t_end, print_info);
    break;
  case ACAS_GLOBAL_CHUNK_COOP:
    res = CAS_noShared_chunk_coop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, coop_lvl, num_hists,
       t_start, t_end, print_info);
    break;
  case ACAS_SHARED_CHUNK_COOP:
    res = CAS_shared_chunk_coop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, coop_lvl, num_hists,
       t_start, t_end, print_info);
    break;
  case ACAS_SHARED_CHUNK_COOP_COL:
    res = CAS_shared_chunk_coop_col<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, coop_lvl, num_hists,
       t_start, t_end, print_info);
    break;

    /* Locking - Exch */
  case AEXCH_GLOBAL_NOCHUNK_FULLCOOP:
    res = exch_noShared_noChunk_fullCoop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz, t_start, t_end,
       print_info);
    break;
  case AEXCH_GLOBAL_CHUNK_FULLCOOP:
    res = exch_noShared_chunk_fullCoop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz, num_threads, seq_chunk,
       t_start, t_end, print_info);
    break;
  case AEXCH_GLOBAL_CHUNK_COOP:
    res = exch_noShared_chunk_coop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, coop_lvl, num_hists,
       t_start, t_end, print_info);
    break;
  case AEXCH_SHARED_CHUNK_COOP:
    res = exch_shared_chunk_coop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, coop_lvl, num_hists,
       t_start, t_end, print_info);
    break;
  case AEXCH_SHARED_CHUNK_COOP_COL:
    res = exch_shared_chunk_coop_col<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, coop_lvl, num_hists,
       t_start, t_end, print_info);
    break;
  case SHLOCK_SHARED_CHUNK_COOP_AEXCH:
    res = exch_shared_chunk_coop_shlock_exch<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, coop_lvl, num_hists,
       t_start, t_end, print_info);
    break;
  case SHLOCK_SHARED_CHUNK_COOP_THREADID:
    res = exch_shared_chunk_coop_shlock_threadid<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, coop_lvl, num_hists,
       t_start, t_end, print_info);
    break;
  case SEQUENTIAL:
    scatter_seq<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz, t_start, t_end);
    res = 0;
    break;
  default:
    res = 1;
    break;
  }
  return res;
}

void find_hardware_details(int print_info) {
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  HWD = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
  BLOCK_SZ = prop.maxThreadsPerBlock;
  SH_MEM_SZ = prop.sharedMemPerBlock;
  if (print_info) {
    printf("Device name: %s\n", prop.name);
    printf("Number of hardware threads: %d\n", HWD);
    printf("Block size: %d\n", BLOCK_SZ);
    printf("Shared memory size: %d\n", SH_MEM_SZ);
    puts("====");
  }
}

int main(int argc, const char* argv[])
{
  /* validate and parse cmd-line arguments */
  int his_sz, kernel, coop_lvl_tmp, n_runs, print_info;
  if(validate_input(argc, argv,
                    &his_sz, &kernel, &coop_lvl_tmp, &n_runs, &print_info) != 0) {
    return -1;
  }

  const char* longest_name = kernel_name(SHLOCK_SHARED_CHUNK_COOP_THREADID);
  const char* name = kernel_name(kernel);
  printf("%s:", name);
  for (unsigned int i = 0; i < strlen(longest_name) - strlen(name); i++) {
    putchar('.');
  }
  putchar(' ');
  if (print_info) {
    printf("\nNumber of runs: %d\n", n_runs);
  }

  find_hardware_details(print_info);

  /* abort as soon as possible */
  unsigned int his_mem_sz = his_sz * sizeof(OUT_T);
  int restrict = (kernel == AADD_SHARED_CHUNK_COOP ||
                  kernel == AADD_SHARED_CHUNK_COOP_COL ||
                  kernel == ACAS_SHARED_CHUNK_COOP ||
                  kernel == ACAS_SHARED_CHUNK_COOP_COL ||
                  kernel == AEXCH_SHARED_CHUNK_COOP ||
                  kernel == AEXCH_SHARED_CHUNK_COOP_COL ||
                  kernel == SHLOCK_SHARED_CHUNK_COOP_AEXCH ||
                  kernel == SHLOCK_SHARED_CHUNK_COOP_THREADID
                  );
  if(restrict && (his_mem_sz > SH_MEM_SZ)) {
    printf("Error: Histogram exceeds shared memory size\n");
    return -1;
  } else if(restrict && coop_lvl_tmp > BLOCK_SZ) {
    printf("Error: Cooperation level exceeds block size\n");
    return -1;
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
  } else if(coop_lvl_tmp == 0 &&
            (kernel == AADD_GLOBAL_CHUNK_COOP ||
             kernel == ACAS_GLOBAL_CHUNK_COOP ||
             kernel == AEXCH_GLOBAL_CHUNK_COOP)) {
    // Heuristic for cooperating in global memory.
    coop_lvl = his_sz;
  } else if(coop_lvl_tmp == 0) {
    // Heuristic for cooperating in local memory.
    coop_lvl = ceil(his_sz / (float) (SH_MEM_SZ / sizeof(OUT_T) / BLOCK_SZ));
    if (kernel == AEXCH_SHARED_CHUNK_COOP ||
        kernel == AEXCH_SHARED_CHUNK_COOP_COL) {
      // The same amount of local memory (assuming ints) is also needed for the
      // lock.
      coop_lvl *= 2;
    }
    // Round up to the nearest power of two to ensure usage of all threads in a block.
    coop_lvl = pow(2, ceil(log(coop_lvl) / log(2)));
  } else {
    coop_lvl = coop_lvl_tmp;
  }

  if ((kernel == SHLOCK_SHARED_CHUNK_COOP_AEXCH ||
       kernel == SHLOCK_SHARED_CHUNK_COOP_THREADID)) {
    if (coop_lvl < (BLOCK_SZ / WARP_SZ) && (BLOCK_SZ / WARP_SZ) % coop_lvl != 0) {
      do {
        coop_lvl++; // can be better
      } while (coop_lvl < (BLOCK_SZ / WARP_SZ) && (BLOCK_SZ / WARP_SZ) % coop_lvl != 0);
    }
  }

  // For global kernels
  int num_hists   = NUM_HISTOS(num_threads, coop_lvl);

  if(print_info) {
    printf("== Heuristic formulas ==\n");
    if(kernel == AADD_GLOBAL_NOCHUNK_FULLCOOP ||
       kernel == ACAS_GLOBAL_NOCHUNK_FULLCOOP ||
       kernel == AEXCH_GLOBAL_NOCHUNK_FULLCOOP) {
      printf("Number of threads:    %d\n", img_sz);
      printf("Sequential chunk:     %d\n", 1);
      printf("Cooperation level:    %d\n", img_sz);
      printf("Number of histograms: %d\n", 1);
    } else if(kernel == AADD_GLOBAL_CHUNK_FULLCOOP ||
              kernel == ACAS_GLOBAL_CHUNK_FULLCOOP ||
              kernel == AEXCH_GLOBAL_CHUNK_FULLCOOP) {
      printf("Number of threads:    %d\n", num_threads);
      printf("Sequential chunk:     %d\n", seq_chunk);
      printf("Cooperation level:    %d\n", num_threads);
      printf("Number of histograms: %d\n", 1);
    } else if(kernel == AADD_GLOBAL_CHUNK_COOP ||
              kernel == ACAS_GLOBAL_CHUNK_COOP ||
              kernel == AEXCH_GLOBAL_CHUNK_COOP) {
      printf("Number of threads:    %d\n", num_threads);
      printf("Sequential chunk:     %d\n", seq_chunk);
      printf("Cooperation level:    %d\n", coop_lvl);
      printf("Number of histograms: %d\n", num_hists);
    } else {
      printf("Cooperation level:    %d\n", coop_lvl);
    }
    printf("====\n");
  }

  /** Kernel versions **/
  int res = 0;
  unsigned long int elapsed, elapsed_total, elapsed_avg;
  int64_t t_start, t_end;

  elapsed_total = 0;
  int n_warmup_runs = 1;
  for (int i = -n_warmup_runs; i < n_runs; i++) {
    if (i < 0) {
      if (print_info) { puts("Warmup run."); }
      res = kernel_run(kernel, h_img, h_his, img_sz, his_sz, num_threads, seq_chunk, coop_lvl, num_hists, &t_start, &t_end, print_info);
      if(res != 0) {
        free(h_img); free(h_his); free(h_seq);
        return res;
      }
      if (i == -n_warmup_runs) {
        /* execute sequential scatter */
        int64_t seq_start, seq_end;
        scatter_seq<MY_OP, IN_T, OUT_T>
          (h_img, h_seq, img_sz, his_sz, &seq_start, &seq_end);

        /* validate the result */
        int valid = validate_array<OUT_T>(h_his, h_seq, his_sz);
        if(!valid) {
          printf("ERROR: Invalid! (coop level: %d)\n", coop_lvl);
          res = -1;
          print_invalid_indices<MY_OP, OUT_T>(h_his, h_seq, his_sz);
          return 1;
        }
      }
    } else {
      if (print_info) { printf("Run %d:\n", i); }
      res = kernel_run(kernel, h_img, h_his, img_sz, his_sz, num_threads, seq_chunk, coop_lvl, num_hists, &t_start, &t_end, print_info);

      if(res != 0) {
        free(h_img); free(h_his); free(h_seq);
        return res;
      }

      /* compute elapsed time for parallel version */
      elapsed = t_end - t_start;
      if (print_info) {
        PRINT_RUNTIME(elapsed);
        printf("\n====\n");
      }
      elapsed_total += elapsed;
    }
  }
  elapsed_avg = elapsed_total / n_runs;
  PRINT_RUNTIME(elapsed_avg);
  printf(" (coop. level: %d)\n", coop_lvl);

  /* free host memory */
  free(h_img); free(h_his); free(h_seq);

  return 0;
}
