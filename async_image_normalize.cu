#include <iostream>

#define FLOAT_MIN -340282346638528859811704183484516925440.0F
#define FLOAT_MAX 340282346638528859811704183484516925440.0F
#define WARP_SIZE 32
#define MAX_STREAMS 32

#define STAGING_SIZE 2000000000

template<unsigned warps_per_block>
__global__ void image_minmax_cu(const float* m, const unsigned size, float* out) {
    
    __shared__ float warp_mins[ warps_per_block ];
    __shared__ float warp_maxs[ warps_per_block ];
    unsigned t_idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    const unsigned stride = gridDim.x * blockDim.x;
    float tmp, tmp_min, local_min = FLOAT_MAX;
    float tmp_max, local_max = FLOAT_MIN;

    for ( ; t_idx < size; t_idx += stride ) {
        tmp = m[t_idx];
        local_min = fminf(local_min, tmp);
        local_max = fmaxf(local_max, tmp);
    }

    #pragma unroll
    for (unsigned offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        tmp_min = __shfl_down_sync(0xffffffff, local_min, offset);
        local_min = fminf(local_min, tmp_min);
        tmp_max = __shfl_up_sync(0xffffffff, local_max, offset);
        local_max = fmaxf(local_max, tmp_max);
    }

    const unsigned lane = threadIdx.x & (WARP_SIZE - 1);
    if ( ! lane ) warp_mins[ threadIdx.x / WARP_SIZE ] = local_min;
    else if ( lane == (WARP_SIZE - 1) ) warp_maxs[ threadIdx.x / WARP_SIZE ] = local_max;
    __syncthreads();

    if ( threadIdx.x < WARP_SIZE ) {
        
        if ( threadIdx.x < warps_per_block ) local_min = warp_mins[ threadIdx.x ];
        else local_min = FLOAT_MAX;

        #pragma unroll
        for (unsigned offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            tmp_min = __shfl_down_sync(0xffffffff, local_min, offset);
            local_min = fminf(local_min, tmp_min);
        }

        if ( ! threadIdx.x ) out[ blockIdx.x ] = local_min;

    } else if ( threadIdx.x < 2*WARP_SIZE ) {

        if ( threadIdx.x - WARP_SIZE < warps_per_block ) local_max = warp_maxs[ threadIdx.x - WARP_SIZE ];
        else local_max = FLOAT_MIN;

        #pragma unroll
        for (unsigned offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            tmp_max = __shfl_down_sync(0xffffffff, local_max, offset);
            local_max = fmaxf(local_max, tmp_max);
        }

        if ( threadIdx.x == WARP_SIZE ) out[ gridDim.x + blockIdx.x ] = local_max;
    }
}

template<unsigned size>
__global__ void warp_minmax_cu(const float* intermediate, float* out) {

    float tmp_min, local_min = FLOAT_MAX;
    float tmp_max, local_max = FLOAT_MIN;

    for (unsigned t_idx = threadIdx.x; t_idx < size; t_idx += WARP_SIZE) {
        local_min = fminf(local_min, intermediate[ t_idx ]);
        local_max = fmaxf(local_max, intermediate[ t_idx + size ]);
    }

    #pragma unroll
    for (unsigned offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        tmp_min = __shfl_down_sync(0xffffffff, local_min, offset);
        local_min = fminf(local_min, tmp_min);
        tmp_max = __shfl_up_sync(0xffffffff, local_max, offset);
        local_max = fmaxf(local_max, tmp_max);
    }

    if ( ! threadIdx.x ) out[0] = local_min;
    else if ( threadIdx.x == (WARP_SIZE-1) ) out[1] = local_max;
}

__global__ void normalize_image_cu(float* img_src, float* img_dest, const unsigned size, const float* min, const float* max) {

    unsigned t_idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    unsigned stride = gridDim.x * blockDim.x;
    const float t_min = *min;
    const float t_max = *max;
    const float t_range = t_max - t_min;

    for ( ; t_idx < size; t_idx += stride ) {
        img_dest[t_idx] = (img_src[t_idx] - t_min) / t_range;
    }
}


void normalize_images(const unsigned* sizes, const unsigned n_images) {
    
    // Allocate host and device memory for staging areas.
    float *host_stage_in, *host_stage_out, *dev_stage_in, *dev_stage_out;
    cudaHostAlloc(&host_stage_in, STAGING_SIZE, cudaHostAllocWriteCombined);
    cudaMalloc(&dev_stage_in, STAGING_SIZE);
    cudaMalloc(&dev_stage_out, STAGING_SIZE);
    cudaHostAlloc(&host_stage_out, STAGING_SIZE, cudaHostAllocDefault);
    
    // Instantiate streams.
    const unsigned n_streams = (n_images < MAX_STREAMS) ? n_images : MAX_STREAMS;
    cudaStream_t streams[ n_streams ];
    for (size_t i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

    //Instantiate events.
    cudaEvent_t device_done_with_stage_in[ n_images ];
    cudaEvent_t host_done_with_stage_out[ n_images ];
    for (size_t i = 0; i < n_images; i++) {
        cudaEventCreate(&device_done_with_stage_in[i]);
        cudaEventCreate(&host_done_with_stage_out[i]);
    }

    constexpr unsigned TPB_x = 512;
    constexpr unsigned BPG_x = 64;
    constexpr unsigned WPB = TPB_x / WARP_SIZE;

    // Allocate device memory for intermediate results.
    float *dev_inter, *dev_minmaxs;
    cudaMalloc(&dev_inter, BPG_x * 2 * n_images * sizeof(float) ); 
    cudaMalloc(&dev_minmaxs, 2 * n_images * sizeof(float) );

    // Move images to device and from device, asynchronously.
    float* local_host_stage_in = host_stage_in;
    float* local_dev_stage_in = dev_stage_in;
    float* local_dev_stage_out = dev_stage_out;
    float* local_host_stage_out = host_stage_out;
    float* local_dev_inter = dev_inter;
    float* local_dev_minmaxs = dev_minmaxs;
    unsigned images_in_prior_stage = 0, images_in_staging = 0;
    size_t image_size, bytes_of_stage_utilized = 0UL;
    cudaStream_t local_stream;
    
    for (size_t i = 0; i < n_images; i++) { 

        local_stream = streams[ i % n_streams ]; 
        image_size = sizes[i];
        bytes_of_stage_utilized += image_size * sizeof(float);

        // If the staging area has been filled.
        if ( bytes_of_stage_utilized > STAGING_SIZE ) {
            local_host_stage_in = host_stage_in;
            local_dev_stage_in = dev_stage_in;
            local_dev_stage_out = dev_stage_out;
            local_host_stage_out = host_stage_out;
            bytes_of_stage_utilized = image_size * sizeof(float);
            images_in_prior_stage = images_in_staging;
            images_in_staging = 0;
        }

        if ( images_in_prior_stage ) cudaEventSynchronize(device_done_with_stage_in[i - images_in_prior_stage]);

        for (size_t j = 0; j < image_size; j++) local_host_stage_in[j] = static_cast<float>(j+1);

        cudaMemcpyAsync(local_dev_stage_in, local_host_stage_in, image_size * sizeof(float), cudaMemcpyHostToDevice, local_stream);
        image_minmax_cu<WPB><<<BPG_x, TPB_x, 0, local_stream>>>(local_dev_stage_in, image_size, local_dev_inter);
        warp_minmax_cu<BPG_x><<<1, WARP_SIZE, 0, local_stream>>>(local_dev_inter, local_dev_minmaxs);
        normalize_image_cu<<<BPG_x/2, TPB_x, 0, local_stream>>>(local_dev_stage_in, local_dev_stage_out, image_size, local_dev_minmaxs, local_dev_minmaxs + 1);
        cudaEventRecord(device_done_with_stage_in[i], local_stream);
        if ( images_in_prior_stage ) cudaEventSynchronize(host_done_with_stage_out[i - images_in_prior_stage]);
        cudaMemcpyAsync(local_host_stage_out, local_dev_stage_out, image_size * sizeof(float), cudaMemcpyDeviceToHost, local_stream);

        // Process / network image from host (asynchronously) here.
        // Note the test code below is not asynchronous.
        cudaStreamSynchronize(local_stream);
        std::cout << "Image " << i << " 1st elem: " << local_host_stage_out[0] << " Last elem: " << local_host_stage_out[image_size-1] << "\n";
        
        cudaEventRecord(host_done_with_stage_out[i], local_stream);

        local_host_stage_in += image_size;
        local_dev_stage_in += image_size;
        local_dev_stage_out += image_size;
        local_host_stage_out += image_size;
        local_dev_inter += BPG_x * 2;
        local_dev_minmaxs += 2;
        images_in_staging++;
    }
    
    // Free host and device memory allocated for staging areas.
    cudaFreeHost(host_stage_in);
    cudaFree(dev_stage_in);
    cudaFree(dev_stage_out);
    cudaFreeHost(host_stage_out);

    // Free device memory allocated for intermediate results.
    cudaFree(dev_inter);
    cudaFree(dev_minmaxs);

    // Destroy events.
    for (size_t i = 0; i < n_images; i++) {
        cudaEventDestroy(device_done_with_stage_in[i]);
        cudaEventDestroy(host_done_with_stage_out[i]);
    }

    // Destroy streams.
    for (size_t i = 0; i < n_streams; i++) cudaStreamDestroy(streams[i]);
}



int main() {

    // Tradition!
    std::cout << "Hello world from CUDA C++!" << std::endl;

    const unsigned n_imgs = 10000;

    unsigned sizes[ n_imgs ];
    for (size_t i = 0; i < n_imgs; i++) sizes[i] = 10000000;
    
    normalize_images(sizes, n_imgs);  
}
