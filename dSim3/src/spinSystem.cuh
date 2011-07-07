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
void copyConstantToDevice(void* device, const void* host, int offset, int size);

void bindCubeCounter(uint* ptr, int size);
void unbindCubeCounter();
void bindTrianglesInCubes(uint* ptr, int size);
void unbindTrianglesInCubes();
//void bindTrgls(uint* ptr, int size);
//void unbindTrgls();
void bindVertices(float* ptr, int size);
void unbindVertices();
void bindTriangleHelpers(float* ptr, int size);
void unbindTriangleHelpers();
void bindRTreeArray(float* ptr, int size);
void unbindRTreeArray();
void bindTreeIndexArray(uint* ptr, int size);
void unbindTreeIndexArray();
void bindTriInfo(uint* ptr, int size);
void unbindTriInfo();

void integrateSystem(
			float* pos,
			uint* randSeed,
			//float* spinInfo,
			spinData* spinInfo,
			float deltaTime,
			float permeability,
			int numBodies,
			float3 magneticGradient,
			float phaseConstant,
			uint iterations, uint* trianglesInCubes, uint* cubeCounter
			);

void integrateSystemVBO(
			uint vboPos,
			uint* randSeed,
			//float* spinInfo,
			spinData* spinInfo,
			float deltaTime,
			float permeability,
			int numBodies,
			float3 magneticGradient,
			float phaseConstant,
			uint iterations, uint* trianglesInCubes, uint* cubeCounter
			);
}
