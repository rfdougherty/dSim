/* 
 * MR Diffusion GPU Device code
 * ----------------------------
 *
 * This file contains the functions (kernels) invoked by the host and to be 
 * executed by the GPU device.
 *
 * Copyright 2008 Bob Dougherty (bobd@stanford.edu) and 
 * Shyamsundar Gopalakrishnan (gshyam@stanford.edu).
 */

#include <cutil.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>

#include "spinKernel.cuh"
#include "spinKernel.cu"
#include "radixsort.cu"

extern "C"
{

void checkCUDA()
{   
#if CUDA2
    CUT_DEVICE_INIT(0,0);
#else
    CUT_DEVICE_INIT();
#endif
}

void allocateArray(void **devPtr, size_t size)
{
    CUDA_SAFE_CALL(cudaMalloc(devPtr, size));
}

void freeArray(void *devPtr)
{
    CUDA_SAFE_CALL(cudaFree(devPtr));
}

void threadSync()
{
    CUDA_SAFE_CALL(cudaThreadSynchronize());
}

void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size)
{   
    if (vbo)
        CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&device, vbo));
    CUDA_SAFE_CALL(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
    if (vbo)
        CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vbo));
}

void copyArrayToDevice(void* device, const void* host, int offset, int size)
{
    CUDA_SAFE_CALL(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
}

void registerGLBufferObject(uint vbo)
{
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(vbo));
}

void unregisterGLBufferObject(uint vbo)
{
    CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(vbo));
}

void bindFiberList(float* ptr, int size)
{
     cudaBindTexture(0,texFiberList,ptr,size*sizeof(float4));     
}

void unbindFiberList()
{
     cudaUnbindTexture(texFiberList);
}

void bindCubeList(uint* ptr, int size)
{
     cudaBindTexture(0,texCubeList,ptr,size*sizeof(uint));     
}

void unbindCubeList()
{
     cudaUnbindTexture(texCubeList);
}


//This is the function that issues batch of threads to the device invoked by the host processor
void integrateSystem(
			         float* pos,
                     uint* randSeed,
                     float deltaTime,
                     float *fiberPos,
                     float permeability,
                     float intraAdc,
                     float extraAdc,
                     float myelinAdc,
                     int numBodies,
                     float3 gradient,
		     float phaseConstant,
                     uint* cubeCounters,
                     uint* cubeList,
                     float cubeLength,
                     uint numCubes,
                     uint maxFibersPerCube,
                     float myelinRadius,
                     uint iterations					
                     )
{

	static bool firstCall = true; //set this to true to display the GPU device specs
	struct cudaDeviceProp devInfo; // Get some information about the device 
	cudaGetDeviceProperties( &devInfo, 0 ); 

	if( firstCall ) { 
	firstCall = false;
	// Write out some info  
	printf("CUDA Device Info:\n\n"); 
	printf("Name: %s\n", devInfo.name ); 
	printf("totalGlobalMem: %u\n",devInfo.totalGlobalMem ); 
	printf("sharedMemPerBlock: %u\n",devInfo.sharedMemPerBlock );
	printf("regsPerBlock: %u\n",devInfo.regsPerBlock ); 
	printf("warpSize: %u\n",devInfo.warpSize );
	printf("memPitch %u\n",devInfo.memPitch );
	printf("maxThreadsPerBlock: %u\n",devInfo.maxThreadsPerBlock); printf("\n\n"); 
	} 

/* TO OPTIMIZE FOR GPU:
   
   1. Run separate kernels for fibers of different orientation. This 
      will eliminate the if statements in the kernel that check for the 
      longitudinal axis. Eg., if all fibers are oriented parallel to z, 
      we will run one kernel and explictly tell it that all fibers are 
      parallel to z. A simple criss-cross patten will involve running 
      twokernels sequentially. This will double the number of times we 
      loop over spins, but I suspect that it will produce a large 
      net increase in speed due to the simplfied kernel with no 
      conditionals.

   2. Sort spins by cubelocation and try to create thread blocks 
      with homogeneous cubelocations. Then, we can load the fibers for 
      that cubeinto shared memory. E.g., see the optimized nbody 
      problem in:
http://www.pas.rochester.edu/~rge21/computing/gpucomputing/cudaoptimise.shtml
      

*/

   int numThreads = min(256, numBodies);
   //int numThreads = min(devInfo.maxThreadsPerBlock, numBodies);
	int numBlocks =  1 +  (numBodies / numThreads); 

	// To avoide extra computation in the kernel, we compute the random walk 
	// standard deviation out here and pass it in.
	// The constant should be 2.0, but this leads to a slight underestimation of the 
	// mean displacement (maybe due to PRNG bias? boundary reflections?) 
    // Try: c=randn(1000000,3).*sqrt(2.0*adc*t);fprintf('%0.6f vs. %0.6f\n',mean(sqrt(sum(c.^2,2))),sqrt(6*adc*t))
    // Given this, a constant of 2.355 would seem to produce the correct displacement
	float intraStdDev = sqrt(2.0f * intraAdc * deltaTime);
	float extraStdDev = sqrt(2.0f * extraAdc * deltaTime);
	float myelinStdDev = sqrt(2.0f * myelinAdc * deltaTime);  

	//float intraStdDev = sqrt(6.0f * intraAdc * deltaTime);
	//float extraStdDev = sqrt(6.0f * extraAdc * deltaTime); 

	// execute the kernel
	integrate<<< numBlocks, numThreads>>>((float4*)pos,
		                                  (uint4*)randSeed,
		                                  deltaTime,
		                                  permeability,
		                                  intraStdDev,
		                                  extraStdDev,
						  myelinStdDev,
		                                  numBodies,
		                                  gradient.x, gradient.y, gradient.z,
		                                  (float4*)fiberPos,
		                                  cubeCounters,
		                                  cubeList,
		                                  phaseConstant,
		                                  cubeLength,
		                                  numCubes,
		                                  maxFibersPerCube,
		                                  myelinRadius*myelinRadius,
		                                  iterations);

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Kernel execution failed");
}

void integrateSystemVbo(
			         uint vboPos,
                     uint* randSeed,
                     float deltaTime,
                     float *fiberPos,
                     float permeability,
                     float intraAdc,
                     float extraAdc,
		     float myelinAdc,
                     int numBodies,
                     float3 gradient,
		     float phaseConstant,
                     uint* cubeCounters,
                     uint* cubeList,
                     float cubeLength,
                     uint numCubes,
                     uint maxFibersPerCube,
                     float myelinRadius,
                     uint iterations					
                     )
{
	float *pos;
	CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&pos, vboPos));
	integrateSystem(pos,randSeed,deltaTime,fiberPos,permeability,intraAdc,extraAdc,myelinAdc,numBodies,gradient,
	                phaseConstant,cubeCounters,cubeList,cubeLength,numCubes,maxFibersPerCube,
	                myelinRadius,iterations);
	//now copy back the space
	CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vboPos));
}


}   // extern "C"
