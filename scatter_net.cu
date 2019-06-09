#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <math.h>
#include <float.h>
#include <complex.h> 

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <errno.h>

#include<cufft.h>
#include <cuComplex.h>
#include <time.h>
     
clock_t start, end;
double gpu_time_used;

//input data holders
const char training_features[] = "data/training_images.bin";
const char training_labels[] = "data/training_labels.bin";
const char filters_imag[] = "data/filters4_imag.bin";
const char filters_real[] = "data/filters4_real.bin";
const char gaussian[] = "data/gaussian.bin";

#define BLOCK_SIZE 1024



void cudaCheckError() {
	cudaError_t e=cudaGetLastError();								
	if(e!=cudaSuccess) {
		printf("Cuda failure %s:%d:'%s'\n",__FILE__ , __LINE__, cudaGetErrorString(e));
		exit(EXIT_FAILURE);							
	}													
}

void printarray(cufftComplex* arr, int size, int ri){
	for(int i=0;i<size;i++){
		if(ri==0)
        		printf("%i ",(int)(arr[i].x));
		if(ri==1)
        		printf("%i ",(int)(arr[i].y));
			
      		if((i+1)%28==0)printf("\n");
        	if((i+1)%(28*28)==0)printf("\n\n\n\n\n");
	}
}
uint64_t getTimeMicroseconds64()
{
        uint64_t        nTime;
        struct timespec tSpec;

        clock_gettime(CLOCK_REALTIME, &tSpec);

        nTime = (uint64_t)tSpec.tv_sec * 1000000 + (uint64_t)tSpec.tv_nsec / 1000;
        return nTime;
}

void memGetInfo(){
        size_t free_byte ;
        size_t total_byte ;

        if ( cudaSuccess != cudaMemGetInfo( &free_byte, &total_byte ) ){
            printf("Error: cudaMemGetInfo fails, %s \n" );
            exit(1);
        }
        double free_db = (double)free_byte ;
        double total_db = (double)total_byte ;
        double used_db = total_db - free_db ;

        printf("GPU memory usage: used = %f, free = %f GB, total = %f GB\n",

            used_db/1024.0/1024.0/1024.0, free_db/1024.0/1024.0/1024.0, total_db/1024.0/1024.0/1024.0);
}


float *get_data(const char *filename, int numOfElements) {
        float *data = (float*) malloc(numOfElements*sizeof(float));
        if (!data) {
                printf("Bad Malloc\n");
                exit(0);
        }
        FILE *ptr = fopen(filename,"rb");

        if (!ptr) {
                printf("Bad file path: %p, %s\n", ptr, strerror(errno));
                exit(0);
        }
        fread(data, sizeof(float), numOfElements,ptr);
        fclose(ptr);

        return data;
}

cufftComplex* load_filters(const char *filters_real,const char *filters_imagine, int filters_size){
	cufftComplex * holder = (cufftComplex*) malloc(filters_size*sizeof(cufftComplex));

	float * tmp = (float*) malloc(filters_size*sizeof(float));
	FILE *ptr = fopen(filters_real,"rb");
	
        if (!ptr) {
                printf("Bad file path: %p, %s\n", ptr, strerror(errno));
                exit(0);
        }
        fread(tmp, sizeof(float), filters_size, ptr);
        fclose(ptr);

	for(int i=0; i<filters_size;i++){
		holder[i].x = tmp[i];
	}

        ptr = fopen(filters_imag,"rb");
          
        if (!ptr) {
                 printf("Bad file path: %p, %s\n", ptr, strerror(errno));                  
	exit(0);
        }
        fread(tmp, sizeof(float), filters_size, ptr);
        fclose(ptr);
 	
	for(int i=0; i<filters_size;i++){
		holder[i].y = tmp[i];
	}
	return holder;
}

cufftComplex* load_Gaussian(const char * gaussian, int size){
	cufftComplex * holder = (cufftComplex*) malloc(size*sizeof(cufftComplex));
	float * tmp = (float*)malloc(size*sizeof(float));
	FILE *ptr = fopen(gaussian,"rb");
        if (!ptr) {
                printf("Bad file path: %p, %s\n", ptr, strerror(errno));
                exit(0);
        }
        fread(tmp, sizeof(float), size, ptr);
        fclose(ptr);
        for(int i=0; i<size;i++){
                holder[i].x = tmp[i];
        }
	return holder;
}

__global__ void duplicate(cufftComplex * from_arr, cufftComplex * to_arr, int from_arr_size, int copies){
	int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
	if (idx < (from_arr_size*copies)){
		to_arr[idx].x = from_arr[idx % from_arr_size].x;
		to_arr[idx].y = from_arr[idx % from_arr_size].y;
	}
	else{
		to_arr[idx].x =0;	
		to_arr[idx].y =0;	
	}
	__syncthreads();
}

__global__ void duplicate2(cufftComplex * signal_ptr_last,cufftComplex * signal_ptr, int signal_size_last, int filters_per_layer){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int signal_size_last_last = signal_size_last / filters_per_layer;
	if(idx < signal_size_last * filters_per_layer){
		signal_ptr[idx] = signal_ptr_last[signal_size_last_last * (idx / signal_size_last) + (idx%(signal_size_last_last))];
	}
	__syncthreads();
}

// dot product with size(v1) = n * size(v2) 
__global__ void dot_unequal_length(cufftComplex *v1, cufftComplex *v2, cufftComplex *v_out, int vec_size1, int vec_size2){
        int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
	if(idx < vec_size1){
		v_out[idx].x = v1[idx].x * v2[(idx)%vec_size2].x - v1[idx].y * v2[(idx)%vec_size2].y;
		v_out[idx].y = v1[idx].x * v2[(idx)%vec_size2].y + v1[idx].y * v2[(idx)%vec_size2].x;
	}
	__syncthreads();
}
// inplace dot product for propagated signal
__global__ void signal_fft_dot_filters_fft(cufftComplex *signal, cufftComplex* filter, int signal_size, int filter_size,  int BATCH){
        int idx = (blockIdx.x) * blockDim.x + threadIdx.x;	
	if(idx < signal_size){
		//int filter_index =( (idx/BATCH) % unit_size )*filter_size + (idx%unit_size)%filter_size;
		int filter_index =  (idx/(BATCH*filter_size)) + idx%(filter_size); // )*filter_size + (idx%unit_size)%filter_size;
		
		signal[idx].x = signal[idx].x * filter[filter_index].x - signal[idx].y * filter[filter_index].y; 
		signal[idx].y = signal[idx].x * filter[filter_index].y + signal[idx].y * filter[filter_index].x;
	}
        __syncthreads();
}


__global__ void modulus(cufftComplex * arr, int arr_size){
	int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
	if (idx < arr_size){
		arr[idx].x = sqrt(arr[idx].x * arr[idx].x + arr[idx].y * arr[idx].y);
		arr[idx].y = 0;
	}
	__syncthreads();
}

__global__ void rescale(cufftComplex* arr, int arr_size, int NX, int NY){
	int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx < arr_size){
		arr[idx].x = arr[idx].x/(NX*NY);	
		arr[idx].y = arr[idx].y/(NX*NY);	
	}
	__syncthreads();			
}

//apply scattering operator(fft+modulus)
//store propagated signals
////1. copy filters_per_layer copies of the original image
////2. perform scattering transform

int main(){

// prepare data
float* training_features_holder, * training_labels_holder;
//data loaded: 60000 of them
int numOfTrainingData = 60000; 
int sizeOfTrainingImage = 28 * 28;
training_features_holder = get_data(training_features, numOfTrainingData * sizeOfTrainingImage);
training_labels_holder = get_data(training_labels, numOfTrainingData);

int numOfLayers = 2;//read from command line
int filters_per_layer = 4;// read from command line
float d_theta = (2 * M_PI)/filters_per_layer;
float * thetas = (float *) malloc(sizeof(float)*filters_per_layer);
for(int i=0;i<filters_per_layer;i++){
	thetas[i] = (i+1)*d_theta;
}


#define NX  56
#define NY  56
#define NRANK  2
#define BATCH  40000

int batchsize;
//int fft_max_batchsize = 500 * pow(filters_per_layer,1);
int fft_max_batchsize = pow(2, 10);//cuFFT optimized for powers of 2

int iters;

//host memory
cufftComplex *filters_h = load_filters(filters_real, filters_imag, sizeof(cufftComplex)*NX*NY*filters_per_layer);

cufftComplex *image_h = (cufftComplex*)malloc(sizeof(cufftComplex)*BATCH*(NX * NY));

cufftComplex *test_holder = (cufftComplex*)malloc(sizeof(cufftComplex)*BATCH*(NX * NY));

cufftComplex *Gaussian_h = load_Gaussian(gaussian, NX*NY);

// array to store final results;
int holders = 0;
int holders_lastlayer = 1;
for(int i=1; i<=numOfLayers;i++){
        holders = holders + holders_lastlayer * filters_per_layer;
        holders_lastlayer = holders;
}
printf("holders needed(NEED TO ENABLE): %i\n",holders);
//cufftComplex *scat_coeff_h = (cufftComplex*)malloc(sizeof(cufftComplex)*BATCH*(NX*NY)*holders);

for(int i=0; i < BATCH * (NX*NY); i++){
	image_h[i].x = training_features_holder[i];
	image_h[i].y = 0;	
}
/*
for(int i=0;i<NX*NY*BATCH;i++){
        printf("%i ",(int)image_h[i].x);
	if(i%28==0)printf("\n");
}
*/
printf("\n");

/*
********************************************
// DEVICE SETUP
********************************************
*/

//mem info
printf("Memory Info before memory allocation:\n");
memGetInfo();

// Create a 2D FFT plan. 
cufftHandle plan_for, plan_inv, plan_test;
int n[NRANK] = {NX, NY};
if (cufftPlanMany(&plan_for, NRANK, n,
				  NULL, 1, 0,
				  NULL, 1, 0,
				  CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT Error: Unable to create for plan\n");
	return 0;	
}

if (cufftPlanMany(&plan_inv, NRANK, n,
				  NULL, 1, 0,
				  NULL, 1, 0,
				  CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT Error: Unable to create inv plan\n");
	return 0;	
}


cufftComplex *filters_d; //holder
cufftComplex *filters_d_fft;// this would be size of pow(filters_per_layer, m)
cufftComplex *image_d; //temp holder
cufftComplex *Gaussian_d;//temp holder
cufftComplex *Gaussian_d_fft;//this would be large;
cufftComplex * prop_signal_d;// this is longest;
//cufftComplex *image_d_fft;

cudaMalloc((void**)&filters_d, sizeof(cufftComplex)*filters_per_layer*NX*NY);
cudaMalloc((void**)&image_d, sizeof(cufftComplex)*BATCH*NX*NY);
/* Two long arrays */
// filters_d_fft scales with pow(filters_per_layer, m)
// should it be this long??
// no!!
//cudaMalloc((void**)&filters_d_fft, sizeof(cufftComplex)*BATCH*NX*NY * pow(filters_per_layer,numOfLayers));

cudaMalloc((void**)&filters_d_fft, sizeof(cufftComplex)*BATCH*NX*NY);

cudaMalloc((void**)&Gaussian_d, sizeof(cufftComplex)*NX*NY);
// should it be this long??
cudaMalloc((void**)&Gaussian_d_fft, sizeof(cufftComplex)*BATCH*NX*NY);
cudaMalloc((void**)&prop_signal_d, sizeof(cufftComplex)*BATCH*(NX*NY)*holders);


if (cudaGetLastError() != cudaSuccess){
	fprintf(stderr, "Cuda error: Failed to allocate\n");
	return 0;	
}

//move filters_h to filter_d
cudaMemcpyAsync(filters_d, filters_h, filters_per_layer*NX*NY*sizeof(cufftComplex),cudaMemcpyHostToDevice);

//move Gaussian_h to Gaussian_d
cudaMemcpyAsync(Gaussian_d, Gaussian_h, NX*NY*sizeof(cufftComplex),cudaMemcpyHostToDevice);

//move image_h to image_d
cudaMemcpyAsync(image_d, image_h, BATCH*NX*NY*sizeof(cufftComplex), cudaMemcpyHostToDevice);

if (cudaGetLastError() != cudaSuccess){
	fprintf(stderr, "Cuda error: Failed to Copy\n");
	return 0;	
}

if (cudaDeviceSynchronize() != cudaSuccess){
  	fprintf(stderr, "Cuda error: Failed to synchronize340\n");
   	return 0;
}
printf("Memory Info after memory allocation:\n");
memGetInfo();
/*
**********************************************************
// Finished preparing filters_d, got image_d
// now START scattering transformation...
// Go down the layers
**********************************************************
*/
//time

double tt = omp_get_wtime();

// Layer 0; convolve with Gaussian Kernel and take modulus
// FFT image_d
if (cufftExecC2C(plan_for, image_d, prop_signal_d, CUFFT_FORWARD) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to execute plan for image\n");
        return 0;         
}
cufftDestroy(plan_for);

if (cufftPlanMany(&plan_for, NRANK, n,
                                  NULL, 1, 0,
                                  NULL, 1, 0,
                                  CUFFT_C2C, 1) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to create for plan\n");
        return 0;
}


// CONVOLUTION

//// FFT Gaussian
if (cufftExecC2C(plan_for, Gaussian_d, Gaussian_d_fft, CUFFT_FORWARD) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to execute plan for filters\n");
        return 0;         
}
if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize3710\n");
        return 0;
}


//// dot product at frequency space;
int Nb = (BATCH*NX*NY+BLOCK_SIZE)/BLOCK_SIZE;
dot_unequal_length<<<Nb, BLOCK_SIZE>>>(prop_signal_d, Gaussian_d_fft, prop_signal_d, NX*NY*BATCH, NX*NY);


if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize371\n");
        return 0;
}

//// INVERSE FFT in-place prop_signal_d
if (cufftExecC2C(plan_inv, prop_signal_d, prop_signal_d, CUFFT_INVERSE) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to execute plan for convoluted image\n");
        return 0;
}

//// Rescale FFT
Nb = (BATCH*NX*NY+BLOCK_SIZE)/BLOCK_SIZE;
rescale<<<Nb,BLOCK_SIZE>>>(prop_signal_d, BATCH*NX*NY, NX, NY);

//// Modulus
Nb = (BATCH*NX*NY+BLOCK_SIZE)/BLOCK_SIZE;
modulus<<<Nb,BLOCK_SIZE>>>(prop_signal_d, BATCH*NX*NY);

// synchronize
if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize391\n");
        return 0;
}

// filters_d_fft
if (cufftPlanMany(&plan_for, NRANK, n,
                                  NULL, 1, 0,
                                  NULL, 1, 0,
                                  CUFFT_C2C, filters_per_layer) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to create plan 418\n");
        return 0;
}

if (cufftExecC2C(plan_for, filters_d, filters_d_fft, CUFFT_FORWARD) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to execute plan for filters\n");
        return 0;
}

if (cudaDeviceSynchronize() != cudaSuccess){
         fprintf(stderr, "Cuda error: Failed to synchronize431\n");
         return 0;
}
cufftDestroy(plan_for);

//for layer k ...
//for(int m=1; m<=numOfLayers;m++){
int signal_pos_last = 0;
cufftComplex * signal_ptr_last;
int signal_size_last;

cufftComplex * signal_ptr;
int signal_size;

// SCATTERING TRANSFROM
int m = 1;
signal_ptr_last = &(prop_signal_d[signal_pos_last]);
signal_size_last = NX * NY * BATCH * pow(filters_per_layer, m-1);
signal_ptr = &(prop_signal_d[signal_pos_last+signal_size_last]);
signal_size = NX * NY * BATCH * pow(filters_per_layer,(m));

// Duplicate
Nb = (signal_size+BLOCK_SIZE)/BLOCK_SIZE;
duplicate<<<Nb, BLOCK_SIZE >>>(signal_ptr_last, signal_ptr, signal_size_last, filters_per_layer);

if (cudaDeviceSynchronize() != cudaSuccess){
         fprintf(stderr, "Cuda error: Failed to synchronize431\n");
         return 0;
}


// FFT prop_signal + DOT PRODUCT
batchsize = (BATCH * (int)(pow(filters_per_layer, m)));
iters = (int)(batchsize/fft_max_batchsize)+1;


if (cufftPlanMany(&plan_for, NRANK, n,
                                          NULL, 1, 0,
                                          NULL, 1, 0,
                                          CUFFT_C2C,  fft_max_batchsize) != CUFFT_SUCCESS){
                fprintf(stderr, "-1 CUFFT Error: Unable to create for plan441\n");
                return 0; 
}  

for(int i=0;i<iters; i++){
	int batch_ith;

        if(i==iters-1){
                batch_ith = batchsize % fft_max_batchsize;
		if (cufftPlanMany(&plan_for, NRANK, n,
                                          NULL, 1, 0,
                                          NULL, 1, 0,
                                          CUFFT_C2C,  batch_ith) != CUFFT_SUCCESS){
                	fprintf(stderr, "%i CUFFT Error: Unable to create for plan441\n",i);
                	return 0;
		}           
	}
        else{   batch_ith = fft_max_batchsize;}

	if (cufftExecC2C(plan_for, &(signal_ptr[i*batch_ith]), &(signal_ptr[i*batch_ith]), CUFFT_FORWARD) != CUFFT_SUCCESS){
        	fprintf(stderr, "CUFFT Error: Unable to execute plan for prop_signal\n");
        	return 0;
	}
	//sync
	if (cudaDeviceSynchronize() != cudaSuccess){
        	fprintf(stderr, "Cuda error: Failed to synchronize 428\n");
        	return 0;
	}
}
cufftDestroy(plan_for);

Nb = (signal_size+BLOCK_SIZE)/BLOCK_SIZE;
signal_fft_dot_filters_fft<<<Nb, BLOCK_SIZE>>>(signal_ptr, filters_d_fft, signal_size, NX*NY, BATCH);
//synchronize
if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize 462\n");
        return 0;
}


// IFFT + MODULUS
batchsize = (BATCH * (int)(pow(filters_per_layer, m)));
iters = (int)(batchsize/fft_max_batchsize)+1;

cufftDestroy(plan_inv);

if (cufftPlanMany(&plan_inv, NRANK, n,
                                          NULL, 1, 0,
                                          NULL, 1, 0,
                                          CUFFT_C2C,  fft_max_batchsize) != CUFFT_SUCCESS){
                fprintf(stderr, "0 CUFFT Error: Unable to create plan 472\n");
                return 0;
}

for(int i=0; i<iters; i++){
	int batch_ith;
	if(i==iters-1){
		batch_ith = batchsize % fft_max_batchsize;
       		if (cufftPlanMany(&plan_inv, NRANK, n,
       	                        	          NULL, 1, 0,
       	                	                  NULL, 1, 0,
       	        	                          CUFFT_C2C,  batch_ith) != CUFFT_SUCCESS){
       	        	fprintf(stderr, "%i CUFFT Error: Unable to create plan 472\n", i);
       	        	return 0;
       		}
	}
	else{	batch_ith = fft_max_batchsize;}

        if (cufftExecC2C(plan_inv, &(signal_ptr[i*batch_ith]), &(signal_ptr[i*batch_ith]), CUFFT_INVERSE) != CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT Error: Unable to execute plan for prop_signal\n");
                return 0;
        }
	//sync
	if (cudaDeviceSynchronize() != cudaSuccess){
                fprintf(stderr, "Cuda error: Failed to synchronize 428\n");
             	return 0;
        }	
}



if (cudaDeviceSynchronize() != cudaSuccess){
         fprintf(stderr, "Cuda error: Failed to synchronize473\n");
         return 0;
}

// Rescale FFT
Nb = (signal_size+BLOCK_SIZE)/BLOCK_SIZE;
rescale<<<Nb,BLOCK_SIZE>>>(signal_ptr, signal_size, NX, NY);

// Modulus
Nb = (signal_size+BLOCK_SIZE)/BLOCK_SIZE;
modulus<<<Nb,BLOCK_SIZE>>>(signal_ptr, signal_size);

//update last pos
signal_pos_last = signal_pos_last + signal_size_last;

for(int m=2; m<=-1;m++){
	
	printf("mem usage before %ith layer\n", m);
	memGetInfo();
	
	signal_ptr_last = &(prop_signal_d[signal_pos_last]);
	signal_size_last = NX * NY * BATCH * pow(filters_per_layer, m-1);
	signal_ptr = &(prop_signal_d[signal_pos_last+signal_size_last]);
	signal_size = NX * NY * BATCH * pow(filters_per_layer,(m));
	
	
	printf("HAHA%i\n", signal_size_last);	
	printf("HAHA%i\n", signal_size);	
	
	Nb = (signal_size+BLOCK_SIZE)/BLOCK_SIZE;
	//duplicate2<<<Nb, BLOCK_SIZE >>>(signal_ptr_last, signal_ptr, signal_size_last, filters_per_layer);
	
	if (cudaDeviceSynchronize() != cudaSuccess){
        	 fprintf(stderr, "Cuda error: Failed to synchronize553\n");
         return 0;
	}
	cufftDestroy(plan_for);		
	// FFT prop_signal + DOT PRODUCT
	batchsize = BATCH * (int)(pow(filters_per_layer,m));
	if (cufftPlanMany(&plan_for, NRANK, n,
	                                  NULL, 1, 0,
	                                  NULL, 1, 0,
	                                  CUFFT_C2C,  batchsize) != CUFFT_SUCCESS){
	        fprintf(stderr, "CUFFT Error: Unable to create plan what the hell\n");
	        return 0;
	}
	
	if (cufftExecC2C(plan_for, signal_ptr, signal_ptr, CUFFT_FORWARD) != CUFFT_SUCCESS){
	        fprintf(stderr, "CUFFT Error: Unable to execute plan for prop_signal\n");
	        return 0;
	}
	cufftDestroy(plan_for);	
	
	// UNIT_SIZE:  sizeof(filters_per_layer) x BATCH
	Nb = (signal_size+BLOCK_SIZE)/BLOCK_SIZE;
	signal_fft_dot_filters_fft<<<Nb, BLOCK_SIZE>>>(signal_ptr, filters_d_fft, signal_size, NX*NY, BATCH);
	
	memGetInfo();
	// IFFT + MODULUS
	if (cufftPlanMany(&plan_inv, NRANK, n,
	                                  NULL, 1, 0,
	                                  NULL, 1, 0,
	                                  CUFFT_C2C,  BATCH * (int)(pow(filters_per_layer,(m)))) != CUFFT_SUCCESS){
	        fprintf(stderr, "CUFFT Error: Unable to create plan\n");
	        return 0;
	}
	
	if (cufftExecC2C(plan_inv, signal_ptr, signal_ptr, CUFFT_INVERSE) != CUFFT_SUCCESS){
	        fprintf(stderr, "CUFFT Error: Unable to execute plan for prop_signal\n");
	        return 0;
	}
	if (cudaDeviceSynchronize() != cudaSuccess){
        	 fprintf(stderr, "Cuda error: Failed to synchronize5530\n");
         	return 0;
	}

	//free memory
	cufftDestroy(plan_inv);
	// Rescale FFT
	Nb = (signal_size+BLOCK_SIZE)/BLOCK_SIZE;
	rescale<<<Nb,BLOCK_SIZE>>>(signal_ptr, signal_size, NX, NY);
	
	// Modulus
	Nb = (signal_size+BLOCK_SIZE)/BLOCK_SIZE;
	modulus<<<Nb,BLOCK_SIZE>>>(signal_ptr, signal_size);
	
	//update last pos
	signal_pos_last = signal_pos_last + signal_size_last;
	                                                     
        printf("mem usage after %ith layer\n", m);
        memGetInfo();

}

//copy back
//cudaMemcpyAsync(test_holder, prop_signal_d, sizeof(cufftComplex)*BATCH*NX*NY, cudaMemcpyDeviceToHost);


double tt1 = omp_get_wtime();

printf("Time Elapsed: %f \n", tt1-tt);

/// test scattering transform
//printf("%f", *(&image_h_fft[1]+1).x);
printf("TEST TEST\n");
//printarray(&(test_holder[BATCH*NX*NY]), BATCH*NX*NY*(filters_per_layer+1), 0);
//printarray((test_holder), BATCH*NX*NY*(filters_per_layer+3), 0);
printf("TEST TEST\n");

cufftDestroy(plan_inv);

cufftDestroy(plan_for);
cudaFree(filters_d); 
cudaFree(filters_d_fft);
cudaFree(image_d);
cudaFree(prop_signal_d); 

 

free(training_features_holder);
free(training_labels_holder);
return 0;

}




