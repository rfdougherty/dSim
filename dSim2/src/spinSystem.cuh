extern "C"
{

void checkCUDA();

void allocateArray(void **devPtr, int size);
void freeArray(void *devPtr);

void threadSync();

void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);
void copyArrayToDevice(void* device, const void* host, int offset, int size);
void registerGLBufferObject(unsigned int vbo);
void unregisterGLBufferObject(unsigned int vbo);

void bindFiberList(float* ptr, int size);
void unbindFiberList();
void bindCubeList(uint* ptr, int size);
void unbindCubeList();
            
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
                     float3 magneticGradient,
                     float phaseConstant,
                     uint* cubeCounters,
                     uint* cubeList,
                     float cubeLength,
                     uint numCubes,
                     uint maxFibersPerCube,
                     float myelinRadius,
                     uint iterations
                     );
                     
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
                     float3 magneticGradient,
                     float phaseConstant,
                     uint* cubeCounters,
                     uint* cubeList,
                     float cubeLength,
                     uint numCubes,
                     uint maxFibersPerCube,
                     float myelinRadius,
                     uint iterations
                     );
}
