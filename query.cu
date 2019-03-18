#include <stdio.h>
#include <cassert>

int main() {
  int n_devices;
  cudaGetDeviceCount(&n_devices);
  assert(n_devices > 0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Device name: %s\n", prop.name);
  printf("Shared memory per block (bytes): %ld\n", prop.sharedMemPerBlock);
  printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
  printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
  printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
  return 0;
}
