#include "ex1.h"

__device__ void create_histogram

__device__ void prefix_sum(int arr[], int arr_size) {
    
    int thIdx = threadIdx.x;
    int inc;
    for(int stride = 1 ; stride < blockDim.x; stride *= 2){
        if(thIdx >= stride){
            inc = arr[ thIdx - stride];
        }    
        __syncthreads();
        if(thIdx >= stride){
            arr[thIdx] += inc;
        }
        __syncthreads();
    }
    return; 
}




__global__ void process_image_kernel(uchar *all_in, uchar *all_out) {
    return; // TODO
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    // TODO define task serial memory buffers
    uchar *gpu_in_img;
    uchar *gpu_out_img;
    int *gpu_g_hist;
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;
    //TODO: allocate GPU memory for a single input image and a single output image
    CUDA_CHECK( cudaMalloc(&context.gpu
_in_img, IMG_HEIGHT * IMG_WIDTH) );
    CUDA_CHECK( cudaMalloc(&context.gpu_out_img, IMG_HEIGHT * IMG_WIDTH) );

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: in a for loop:
    for(int i = 0 ; i < N_IMAGES ; i++){
        //   1. copy the relevant image from images_in to the GPU memory you allocated
        CUDA_CHECK( cudaMemcpy(context->gpu
    _in_img , (images_in + i * IMG_HEIGHT * IMG_WIDTH) , IMG_HEIGHT * IMG_WIDTH, cudaMemcpyHostToDevice);
       
        //   2. invoke GPU kernel on this image  
        ////////////TBD how many blocks and threads should we use?
        process_image_kernel<<1,1>>(context->gpu_in_img , context->_out_img);
        ////////////

        //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
        CUDA_CHECK( cudaMemcpy((images_out + i * IMG_HEIGHT * IMG_WIDTH) , context->gpu_out_img , IMG_HEIGHT * IMG_WIDTH, cudaMemcpyDevicetToHost);
    }
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //TODO: free resources allocated in task_serial_init
    CUDA_CHECK(cudaFree(context->gpu_in_img);
    CUDA_CHECK(cudaFree(context->gpu_out_img);
    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // TODO define bulk-GPU memory buffers
};

/* Allocate GPU memory for all the input and output images.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    //TODO: allocate GPU memory for a all input images and all output images

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //TODO: free resources allocated in gpu_bulk_init

    free(context);
}
