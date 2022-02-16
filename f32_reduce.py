import numpy as np
import numba as nb
from numba import cuda
import time

TPB_X = 1024


@cuda.jit
def reduce_add_CU(X, out):

  n = len(X)
  i = (cuda.blockDim.x * cuda.blockIdx.x) + cuda.threadIdx.x
  
  X_shared = cuda.shared.array(TPB_X, nb.float32)

  if i < n:
    X_shared[cuda.threadIdx.x] = X[i]
  else:
    X_shared[cuda.threadIdx.x] = 0.0

  cuda.syncthreads()

  sum_block = cuda.blockDim.x // 2
  while sum_block >= 1:
    
    if cuda.threadIdx.x < sum_block:
      X_shared[cuda.threadIdx.x] += X_shared[cuda.threadIdx.x + sum_block]
    
    sum_block //= 2
    cuda.syncthreads()

  if cuda.threadIdx.x == 0:
    cuda.atomic.add(out, 0, X_shared[0])

  
def f32_reduce(X, report_duration = False):

  X_d = cuda.to_device( X.astype(np.float32) )
  
  # Unsure how to initialize a device array to a zero in numba.
  # As such, I suboptimally initialize the array on the host and transfer.
  sum = np.zeros(1, dtype = np.float32)
  sum_d = cuda.to_device(sum)

  bpg_x = (len(X) + TPB_X - 1) // TPB_X

  if report_duration:
    start = time.perf_counter()
    reduce_add_CU[bpg_x, TPB_X](X_d, sum_d)
    end = time.perf_counter()
    print(f"f32_reduce kernel execution duration: {end - start}")
  else:
    reduce_add_CU[bpg_x, TPB_X](X_d, sum_d)

  sum_d.copy_to_host(sum)

  return sum[0]



import time

X = np.ones(1000000000, dtype = np.float32)

start = time.perf_counter()
sum_numpy = X.sum()
end = time.perf_counter()
print(f"Numpy sum: {sum_numpy}   Duration: {end - start}")

start = time.perf_counter()
sum_CUDA = f32_reduce(X, report_duration = True)
end = time.perf_counter()
print(f"CUDA sum: {sum_CUDA}   Duration: {end - start}")

