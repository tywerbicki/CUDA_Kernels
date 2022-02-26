#include <iostream>

const unsigned TPB_x = 512;
const unsigned WARP_SIZE = 32;
const unsigned WPB = TPB_x / WARP_SIZE;

__global__ void curve_lengths_cu(const float *data, const int n_sig, const int len, float *curve_lengths) {

    __shared__ float warp_sums[WPB];
    __shared__ float raw[TPB_x];
    const unsigned laneIdx = threadIdx.x & (WARP_SIZE - 1);
    unsigned mask, offset;
    float elem, tmp, curve_length = 0.0F;
    int pairs_processed, pairs_to_process;

    for (int current_signal = blockIdx.x; current_signal < n_sig; current_signal += gridDim.x) {
        
        pairs_processed = 0;
        pairs_to_process = min(len - 1, blockDim.x - 1);

        while ( pairs_to_process > 0 ) {    

            if ( threadIdx.x < pairs_to_process + 1 ) {
                elem = data[ current_signal*len + pairs_processed + threadIdx.x];
                raw[threadIdx.x] = elem;
            } 
            __syncthreads();

            if ( threadIdx.x < pairs_to_process ) {
                elem = raw[threadIdx.x + 1] - elem;
                elem = sqrtf( 1 + elem*elem );
            } else elem = 0.0F;
            __syncthreads();

            for (offset = WARP_SIZE >> 1; offset > 0; offset>>=1 )
                elem += __shfl_down_sync(0xffffffff, elem, offset);

            if ( ! laneIdx ) warp_sums[ threadIdx.x / WARP_SIZE ] = elem;
            __syncthreads();

            offset = WPB >> 1;
            mask = __ballot_sync(0xffffffff, threadIdx.x < offset);
            if ( threadIdx.x < offset ) {
                for (; offset > 0; offset>>=1) {
                    tmp = warp_sums[ threadIdx.x + offset ];
                    __syncwarp(mask);
                    warp_sums[threadIdx.x] += tmp;     
                    __syncwarp(mask);               
                }
            }

            if ( threadIdx.x == 0 ) curve_length += warp_sums[0];
            pairs_processed += pairs_to_process;
            pairs_to_process = min(len - 1 - pairs_processed, blockDim.x - 1);
        } 

        if ( threadIdx.x == 0 ) curve_lengths[current_signal] = curve_length;
    }
}


float* curve_lengths(const float* data, const int n_sig, const int len) {

    const size_t data_n_bytes = n_sig * len * sizeof(float);
    float *data_D;
    cudaMalloc(&data_D, data_n_bytes);
    cudaMemcpy(data_D, data, data_n_bytes, cudaMemcpyHostToDevice);

    const size_t cl_n_bytes = n_sig * sizeof(float);
    float *curve_lengths, *curve_lengths_D;
    curve_lengths = new float[cl_n_bytes];
    cudaMalloc(&curve_lengths_D, cl_n_bytes);

    const int BPG_x = (n_sig < 65000) ? n_sig : 65000;
    curve_lengths_cu<<<BPG_x, TPB_x>>>(data_D, n_sig, len, curve_lengths_D);

    cudaMemcpy(curve_lengths, curve_lengths_D, cl_n_bytes, cudaMemcpyDeviceToHost);
    cudaFree(data_D);
    cudaFree(curve_lengths_D);

    return curve_lengths;
}




int main() {

    const int n_sig = 5, len = 5000;
    float* data = new float[n_sig*len];

    for (size_t i = 0; i < n_sig; i++) {
        for (size_t j = 0; j < len; j++) {
            *(data + i*len + j) = (float)j;
        }  
    }

    float *cls = curve_lengths(data, n_sig, len); 
    
    for (size_t i = 0; i < n_sig; i++) {
        std::cout << cls[i] << " ";
    }
    std::cout << std::endl;
    
    delete[] data;
    delete[] cls;
}
