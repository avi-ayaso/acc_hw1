#include "ex1.h"

#define VEC_SIZE 256
#define HISTOGRAM_SIZE 256


__device__ void prefix_sum(int *arr , int len) {
    
    int thIdx = threadIdx.x;
    int inc;
    int idx;
    
    for(int stride = 1 ; stride < len ; stride *= 2){
                
        if(thIdx >= stride &&  thiDx < len){
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

__device__ void map_calc(int *map ,int *cdf, int idx){
    float map_value;
    if(idx < HISTOGRAM_SIZE){
        map_value = float(cdf[idx]) / (IMG_HEIGHT * IMG_WIDTH); 
        map[idx] = ((uchar)(N_COLORS * map_value)) * (256 / N_COLORS);
    }
}

__global__ void process_image_kernel(uchar *all_in, uchar *all_out) {
    
    __shared__ int histogram[HISTOGRAM_SIZE];
    int *cdf = histogram; // we dont need more than one shared array
    int *map = histogram; // we dont need more than one shared array
    int thIdx = threadIdx.x;
    int blIdx = blockIdx.x;
    int blockSize = blockDim.x;
    int chunkSize = (IMG_HEIGHT * IMG_WIDTH) / blockSize;
    int internal_offset = thIdx * chunkSize;
    int image_offset = blIdx * IMG_HEIGHT * IMG_WIDTH;
    int image_end = (blIdx + 1) * IMG_HEIGHT * IMG_WIDTH;
    int offset = image_offset + internal_offset;
    int i;

    //zeroize histogram in parallel
    for(i = thIdx; i < HISTOGRAM_SIZE; i += blockSize){
        histogram[i] = 0;
    }
    __syncthreads();
    
    //build histogram
    for(i = 0 ; i < chunkSize && ((i + offset) < image_end) ; i++){
        atomicAdd(&histogram[all_in[i + offset]], 1);
    }
    __syncthreads();
    
    //get the cdf
    prefix_sum(cdf , HISTOGRAM_SIZE);

    //get the new color quantizer
    map_calc(map ,cdf,thIdx);
    __syncthreads();
    
    //create new image
    for(i = 0 ; i < chunkSize && ((i + offset) < image_end) ; i++){
        all_out[i + offset] = map[all_in[i + offset]];
    }
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    // TODO define task serial memory buffers
    uchar *gpu_in_img;
    uchar *gpu_out_img;
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;
    //allocate GPU memory for a single input image and a single output image
    CUDA_CHECK( cudaMalloc(&context->gpu_in_img, IMG_HEIGHT * IMG_WIDTH) );
    CUDA_CHECK( cudaMalloc(&context->gpu_out_img, IMG_HEIGHT * IMG_WIDTH) );

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: in a for loop:
    int offset;
    for(int i = 0 ; i < N_IMAGES ; i++){
        //   1. copy the relevant image from images_in to the GPU memory you allocated
        offset = i * IMG_HEIGHT * IMG_WIDTH ;
        CUDA_CHECK( cudaMemcpy(context->gpu_in_img , images_in + offset , IMG_HEIGHT * IMG_WIDTH, cudaMemcpyHostToDevice) );
       
        //   2. invoke GPU kernel on this image  
        process_image_kernel<<<1 , 1024>>>(context->gpu_in_img , context->gpu_out_img);

        //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
        CUDA_CHECK( cudaMemcpy(images_out + offset , context->gpu_out_img , IMG_HEIGHT * IMG_WIDTH, cudaMemcpyDeviceToHost) );
    }
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //TODO: free resources allocated in task_serial_init
    CUDA_CHECK( cudaFree(context->gpu_in_img) );
    CUDA_CHECK( cudaFree(context->gpu_out_img) );
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
