import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

dev = cuda.Device(0)
devAttributes = dev.get_attributes()

for attribute, datum in devAttributes.items():
  
  if str(attribute) == 'MAX_BLOCK_DIM_X':
    MAX_BLOCK_DIM_X = int(str(datum))
  elif str(attribute) == 'MAX_GRID_DIM_X':
    MAX_GRID_DIM_X = int(str(datum))

mod = SourceModule(
  """
  __device__
  float Rho(const unsigned int nrow, const float sum_X, const float sum_Y,
                const float sum_XY, const float squareSum_X, const float squareSum_Y)
  {
    return (nrow*sum_XY - sum_X*sum_Y ) /
           sqrtf( (nrow*squareSum_X - sum_X*sum_X) * (nrow*squareSum_Y - sum_Y*sum_Y) );
  }

  __global__
  void GetCorrFlag(float* mat, float* targ, unsigned int* out, const unsigned int nrow,  
                const unsigned int ncol, const float sum_Y, const float squareSum_Y,
                const float low, const float high)
  {
      unsigned int j = (blockIdx.x * blockDim.x) + threadIdx.x;
      const unsigned int s_x = blockDim.x * gridDim.x;
      float sum_X, squareSum_X, sum_XY, tmp;
      
      while (j < ncol)
      {
          sum_X = 0; squareSum_X = 0; sum_XY = 0;
          for (size_t i = 0; i < nrow; i++)  
          {
              tmp = *(mat + (i*ncol) + j);
              sum_X += tmp;
              squareSum_X += tmp * tmp;
              sum_XY += tmp * targ[i];
          }
          float rho = Rho(nrow, sum_X, sum_Y, sum_XY, squareSum_X, squareSum_Y);
          if (rho > low && rho < high)
          { out[j] = 1; } 
          else
          { out[j] = 0; }
          j += s_x;
      }
  }
  """
  )

nrow = 100 ; ncol = 3500000
a = np.random.normal(size = nrow * ncol).reshape(nrow, ncol).astype(np.float32)
a_d = cuda.mem_alloc(a.nbytes)
b = (np.arange(nrow)*ncol).astype(np.float32)
sum_Y = b.sum() ; squareSum_Y = (b * b).sum()
b_d = cuda.mem_alloc(b.nbytes)
c = np.empty(ncol).astype(np.int32)
c_d = cuda.mem_alloc(c.nbytes)
cuda.memcpy_htod(a_d, a)
cuda.memcpy_htod(b_d, b)

tpb_x = MAX_BLOCK_DIM_X
optimalBlocks_x = (ncol + tpb_x - 1) // tpb_x
nb_x = int(np.minimum(optimalBlocks_x, MAX_GRID_DIM_X))

GetCorrFlag = mod.get_function("GetCorrFlag")
GetCorrFlag(
         a_d, b_d, c_d, np.int32(nrow), np.int32(ncol),
         sum_Y, squareSum_Y, np.float32(.2), np.float32(.8),
         grid = (nb_x, 1, 1), block = (tpb_x, 1, 1)
         )

cuda.memcpy_dtoh(c, c_d)

#Use the integer mask here as desired.
#Operations containing the vector 'c'
#...
#...
