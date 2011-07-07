/////////////////////////////////////////////////////////////////////////////////////////
// File name:		spinSystem.cu
// Description:		Definition of all CUDA functions that are not used inside the
//			kernel.
/////////////////////////////////////////////////////////////////////////////////////////

#include <cutil.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>

#include "spinKernel.cu"
#include "radixsort.cu"
//#include "dSimDataTypes.h"

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


///////////////////////////////////////////////////////////////////////
// Function name:	allocateArray
// Description:		Allocate memory on device for an array pointed to
//			by devPtr of size size.
///////////////////////////////////////////////////////////////////////
void allocateArray(void **devPtr, size_t size)
{
	CUDA_SAFE_CALL(cudaMalloc(devPtr,size));
}


///////////////////////////////////////////////////////////////////////
// Function name:	freeArray
// Description:		Free up the device memory used by the array pointed
//			to by devPtr
///////////////////////////////////////////////////////////////////////
void freeArray(void *devPtr)
{
	CUDA_SAFE_CALL(cudaFree(devPtr));
}


///////////////////////////////////////////////////////////////////////
// Function name:	threadSync
// Description:		Block until the device has completed all preceding
//			requested tasks.
///////////////////////////////////////////////////////////////////////
void threadSync()
{
	CUDA_SAFE_CALL(cudaThreadSynchronize());
}


///////////////////////////////////////////////////////////////////////
// Function name:	copyArrayFromDevice
// Description:		Copy array from device (pointed to by device parameter)
//			to array on host (pointed to by host parameter)
///////////////////////////////////////////////////////////////////////
void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size)
{
	if (vbo)
		CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&device, vbo));
	CUDA_SAFE_CALL(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
	if (vbo)
		CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vbo));
}


////////////////////////////////////////////////////////////////////////
// Function name:	copyArrayToDevice
// Description:		Copy array from host (pointed to by host parameter)
//			to array on device (pointed to by device parameter)
////////////////////////////////////////////////////////////////////////
void copyArrayToDevice(void* device, const void* host, int offset, int size)
{
	CUDA_SAFE_CALL(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
}


/////////////////////////////////////////////////////////////////////////
// Function name:	copyConstantToDevice
// Description:		Copy constant from host (with name host) to device
//			(with name device).
/////////////////////////////////////////////////////////////////////////
void copyConstantToDevice(void* device, const void* host, int offset, int size)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol((char *) device, host, size));
}


//////////////////////////////////////////////////////////////////////////
// Function name:	registerGLBufferObject
// Description:		Registers the buffer object of ID vbo for access by CUDA.
//////////////////////////////////////////////////////////////////////////
void registerGLBufferObject(uint vbo)
{
	CUDA_SAFE_CALL(cudaGLRegisterBufferObject(vbo));
}


//////////////////////////////////////////////////////////////////////////
// Function name:	unregisterGLBufferObject
// Description:		Unregisters the buffer object of ID vbo for access by CUDA
//			and releases any CUDA resources associated with the buffer.
//////////////////////////////////////////////////////////////////////////
void unregisterGLBufferObject(uint vbo)
{
	CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(vbo));
}


//////////////////////////////////////////////////////////////////////////
// The following functions bind/unbind various arrays from host to device
// texture memory.
// Note: Should combine into one function
//////////////////////////////////////////////////////////////////////////
void bindCubeCounter(uint* ptr, int size)						// Test
{
	cudaBindTexture(0,texCubeCounter,ptr,size*sizeof(uint));
}

void unbindCubeCounter()								// Test
{
	cudaUnbindTexture(texCubeCounter);
}

void bindTrianglesInCubes(uint* ptr, int size)						// Test
{
	cudaBindTexture(0,texTrianglesInCubes,ptr,size*sizeof(uint));
}

void unbindTrianglesInCubes()								// Test
{
	cudaUnbindTexture(texTrianglesInCubes);
}
/*
void bindTrgls(uint* ptr, int size)							// Test
{
	cudaBindTexture(0,texTrgls,ptr,size*sizeof(uint));
}

void unbindTrgls()									// Test
{
	cudaUnbindTexture(texTrgls);
}
*/
void bindVertices(float* ptr, int size)							// Test
{
	if (size>0){
		cudaBindTexture(0,texVertices,ptr,size*sizeof(float));
	}
}

void unbindVertices()									// Test
{
	cudaUnbindTexture(texVertices);
}

void bindTriangleHelpers(float* ptr, int size)						// Test
{
	if (size>0){
		cudaBindTexture(0,texTriangleHelpers,ptr,size*sizeof(float));
	}
}

void unbindTriangleHelpers()								// Test
{
	cudaUnbindTexture(texTriangleHelpers);
}

void bindRTreeArray(float* ptr, int size)						// Test
{
	if (size>0){
		cudaBindTexture(0,texRTreeArray,ptr,size*sizeof(float));
	}
}

void unbindRTreeArray()									// Test
{
	cudaUnbindTexture(texRTreeArray);
}

void bindTreeIndexArray(uint* ptr, int size)						// Test
{
	if (size>0){
		cudaBindTexture(0,texCombinedTreeIndex,ptr,size*sizeof(uint));
	}
}

void unbindTreeIndexArray()								// Test
{
	cudaUnbindTexture(texCombinedTreeIndex);
}

void bindTriInfo(uint* ptr, int size)						// Test
{
	if (size>0){
		cudaBindTexture(0,texTriInfo,ptr,size*sizeof(uint));
	}
}

void unbindTriInfo()								// Test
{
	cudaUnbindTexture(texTriInfo);
}

///////////////////////////////////////////////////////////////////////////
// Function name:	integrateSystem
// Description:		Run the kernel for spin computations
///////////////////////////////////////////////////////////////////////////
void integrateSystem(
			float* pos,
			uint* randSeed,
			//float* spinInfo,
			spinData* spinInfo,
			float deltaTime,
			float permeability,
			int numBodies,
			float3 gradient,
			float phaseConstant,
			uint iterations, uint* trianglesInCubes, uint* cubeCounter
			)
{
	static bool firstCall = true;
	struct cudaDeviceProp devInfo;
	cudaGetDeviceProperties(&devInfo,0);

	if (firstCall){
		firstCall = false;
		// Write out some info
		printf("CUDA device info:\n\n");
		printf("Name: %s\n", devInfo.name);
		printf("totalGlobalMem: %u\n", devInfo.totalGlobalMem);
		printf("sharedMemPerBlock: %u\n", devInfo.sharedMemPerBlock);
		printf("regsPerBlock: %u\n", devInfo.regsPerBlock);
		printf("warpSize: %u\n", devInfo.warpSize);
		printf("memPitch: %u\n", devInfo.memPitch);
		printf("maxThreadsPerBlock: %u\n", devInfo.maxThreadsPerBlock);
		printf("\n\n");
	}

	// Number of threads will normally be 128
	int numThreads = min(128, numBodies);
	int numBlocks = 1 + numBodies/numThreads;
	
	// Execute the kernel
	integrate<<< numBlocks, numThreads >>>( 
						(float3*) pos,
						(uint2*) randSeed,
						//(float4*) spinInfo,
						spinInfo,
						deltaTime,
						permeability,
						numBodies,
						gradient.x, gradient.y, gradient.z,
						phaseConstant,
						iterations, trianglesInCubes, cubeCounter);

	CUT_CHECK_ERROR("Kernel execution failed\n");
}


//////////////////////////////////////////////////////////////////////////////////////
// Function name:	integrateSystemVBO
// Description:		Register the vertex buffer object for access by CUDA, perform
//			the GPU computation using integrateSystem, then unregister
//			the VBO.
//////////////////////////////////////////////////////////////////////////////////////
void integrateSystemVBO(
			uint vboPos,
			uint* randSeed,
			//float* spinInfo,
			spinData* spinInfo,
			float deltaTime,
			float permeability,
			int numBodies,
			float3 gradient,
			float phaseConstant,
			uint iterations, uint* trianglesInCubes, uint* cubeCounter
			)
{
	float *pos;
	CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&pos, vboPos));
	integrateSystem(pos,randSeed,spinInfo,deltaTime,permeability, numBodies, gradient, phaseConstant, iterations, trianglesInCubes, cubeCounter);
	CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vboPos));
}


} // extern "C"
