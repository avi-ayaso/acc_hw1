#include "ex1.h"

#define VEC_SIZE 256
#define HISTOGRAM_SIZE 256




__device__ void prefix_sum(int *arr , int len) {
    
    int thIdx = threadIdx.x;
    int inc;
    
    for(int stride = 1 ; stride < len ; stride *= 2){
                
        if(thIdx >= stride &&  thIdx < len){
            inc = arr[thIdx - stride];
        }    
        __syncthreads();
        if(thIdx >= stride && thIdx < len){
            arr[thIdx] += inc;
        }
        __syncthreads();
    }
    return; 
}

__device__ void map_calc(uchar *map ,int *cdf, int idx){
    float map_value = IMG_HEIGHT * IMG_WIDTH;
    if(idx < HISTOGRAM_SIZE){
        map[idx] = ((uchar)(N_COLORS * (cdf[idx] /map_value))) * (256 / N_COLORS);
    }
}

__global__ void process_image_kernel(uchar *all_in, uchar *all_out){
    int thIdx = threadIdx.x;
    int offset = IMG_WIDTH * IMG_HEIGHT * blockIdx.x + thIdx;
    __shared__ int histogram[HISTOGRAM_SIZE];
    __shared__ uchar map[HISTOGRAM_SIZE];
    int * cdf = histogram;
    if (thIdx < HISTOGRAM_SIZE) {
        histogram[thIdx] = 0;
    }
    __syncthreads();
    for(int j = 0; j < IMG_WIDTH * IMG_HEIGHT; j += blockDim.x){
        int pixelValue = all_in[offset + j];
        atomicAdd(histogram + pixelValue, 1);
    }
    __syncthreads();
    prefix_sum(histogram, HISTOGRAM_SIZE);
    map_calc(map ,cdf,thIdx);
    __syncthreads();
    for(int j = 0; j < IMG_WIDTH * IMG_HEIGHT; j += blockDim.x){
        int pixelValue = all_in[offset + j];
        all_out[offset + j] = map[pixelValue];
    }
    return;
}


/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    // TODO define task serial memory buffers
    uchar *gpu_in_img[N_IMAGES];
    uchar *gpu_out_img[N_IMAGES];
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;
    //allocate GPU memory for a single input image and a single output image
    for(int i = 0 ; i < N_IMAGES ; i++){
        CUDA_CHECK( cudaMalloc(&context->gpu_in_img[i], IMG_HEIGHT * IMG_WIDTH) );
        CUDA_CHECK( cudaMalloc(&context->gpu_out_img[i], IMG_HEIGHT * IMG_WIDTH) );
    }
    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{
    
    //TODO: in a for loop:
    int offset = 0;
    for(int i = 0 ; i < N_IMAGES ; i++ , offset += IMG_HEIGHT * IMG_WIDTH){
        //   1. copy the relevant image from images_in to the GPU memory you allocated
        // offset = i * IMG_HEIGHT * IMG_WIDTH ;
        CUDA_CHECK( cudaMemcpy(context->gpu_in_img[i] , images_in + offset , IMG_HEIGHT * IMG_WIDTH, cudaMemcpyHostToDevice) );
        //   2. invoke GPU kernel on this image  
        process_image_kernel<<<1 , 1024>>>(context->gpu_in_img[i] , context->gpu_out_img[i]);
        //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
        CUDA_CHECK( cudaMemcpy(images_out + offset , context->gpu_out_img[i] , IMG_HEIGHT * IMG_WIDTH, cudaMemcpyDeviceToHost) );
    }
    CUDA_CHECK( cudaDeviceSynchronize() );
    
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //TODO: free resources allocated in task_serial_init
    for(int i = 0 ; i < N_IMAGES ; i++){
        CUDA_CHECK( cudaFree(context->gpu_in_img[i]) );
        CUDA_CHECK( cudaFree(context->gpu_out_img[i]) );
    }
    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // TODO define bulk-GPU memory buffers
    uchar *gpu_in_imgs;
    uchar *gpu_out_imgs;
};

/* Allocate GPU memory for all the input and output images.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    //TODO: allocate GPU memory for a all input images and all output images
    CUDA_CHECK( cudaMalloc(&context->gpu_in_imgs, N_IMAGES * IMG_HEIGHT * IMG_WIDTH) );
    CUDA_CHECK( cudaMalloc(&context->gpu_out_imgs, N_IMAGES * IMG_HEIGHT * IMG_WIDTH) );
    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    //   1. copy all input images from images_in to the GPU memory you allocated
    CUDA_CHECK( cudaMemcpy(context->gpu_in_imgs , images_in, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, cudaMemcpyHostToDevice) );

    //   2. invoke a kernel with N_IMAGES threadblocks, each working on a different image
    process_image_kernel<<<N_IMAGES , 1024>>>(context->gpu_in_imgs , context->gpu_out_imgs);

    //   3. copy output images from GPU memory to images_out
    CUDA_CHECK( cudaMemcpy(images_out, context->gpu_out_imgs , N_IMAGES * IMG_HEIGHT * IMG_WIDTH, cudaMemcpyDeviceToHost) );

}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //free resources allocated in gpu_bulk_init
    CUDA_CHECK(cudaFree(context->gpu_in_imgs));
    CUDA_CHECK(cudaFree(context->gpu_out_imgs));
    free(context);

}
