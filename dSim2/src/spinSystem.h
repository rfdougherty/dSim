#ifndef __BODYSYSTEMCUDA_H__
#define __BODYSYSTEMCUDA_H__
#include <fstream>
#define DEBUG_GRID 0
#define DO_TIMING 0

#include "options.h"
#include <math.h>

typedef unsigned int uint;

typedef struct _float3
{
  float x;
  float y;
  float z;
}float3;

typedef struct _uint4
{
  uint x;
  uint y;
  uint z;
  uint w;
}uint4;

typedef struct _uint2
{
	uint a;
	uint b;
}uint2;



// Some simple vector ops for float3's, to mimic those found in cudautil_math.h.
inline float3 make_float3(float x, float y, float z){
   float3 p;
   p.x = x; p.y = y; p.z = z;
   return p;
}
// addition
inline float3 operator+(float3 a, float3 b){
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline float3 operator+(float3 a, float s){
    return make_float3(a.x + s, a.y + s, a.z + s);
}
inline float3 operator+(float s, float3 a){
    return make_float3(a.x + s, a.y + s, a.z + s);
}
inline void operator+=(float3 &a, float3 b){
    a.x += b.x; a.y += b.y; a.z += b.z;
}
// subtract
inline float3 operator-(float3 a, float3 b){
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline void operator-=(float3 &a, float3 b){
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
// multiply
inline float3 operator*(float3 a, float3 b){
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline float3 operator*(float3 a, float s){
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline float3 operator*(float s, float3 a){
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline void operator*=(float3 &a, float s){
    a.x *= s; a.y *= s; a.z *= s;
}
// divide
inline float3 operator/(float3 a, float3 b){
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline float3 operator/(float3 a, float s){
    float inv = 1.0f / s;
    return a * inv;
}
inline float3 operator/(float s, float3 a){
    float inv = 1.0f / s;
    return a * inv;
}
inline void operator/=(float3 &a, float s){
    float inv = 1.0f / s;
    a *= inv;
}
// dot product
inline float dot(float3 a, float3 b){ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
// cross product
inline float3 cross(float3 a, float3 b){ 
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}
// length
inline float length(float3 v){
    return sqrtf(dot(v, v));
}
// floor
inline float3 floor(const float3 v){
    return make_float3(floor(v.x), floor(v.y), floor(v.z));
}
// min
inline float3 fminf(float3 a, float3 b){
   return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
inline float3 fminf(float s, float3 a){
   return make_float3(fminf(a.x,s), fminf(a.y,s), fminf(a.z,s));
}
inline float3 fminf(float3 a, float s){
   return make_float3(fminf(a.x,s), fminf(a.y,s), fminf(a.z,s));
}
// max
inline float3 fmaxf(float3 a, float3 b){
   return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
inline float3 fmaxf(float3 a, float s){
   return make_float3(fmaxf(a.x,s), fmaxf(a.y,s), fmaxf(a.z,s));
}
inline float3 fmaxf(float s, float3 a){
   return make_float3(fmaxf(a.x,s), fmaxf(a.y,s), fmaxf(a.z,s));
}

typedef struct _int3
{
  int x;
  int y;
  int z;
}int3;

// CUDA SpinSystem definition: runs on the GPU
class SpinSystem
{
public:
	SpinSystem(int numSpins, bool useGpu, float spaceScale, float gyromagneticRatio, bool useVbo);
	~SpinSystem();

	bool build();

	bool initFibers(float fiberRadius, float fiberRadiusStd, float fiberSpace, float fiberSpaceStd, float fiberInnerRadiusProportion, float fiberCrossProportion);

   	bool initFibers(FILE *fiberFilePtr, float fiberInnerRadiusProportion);

	enum SpinConfig
	{
	 CONFIG_RANDOM,
	 CONFIG_GRID,
	 _NUM_CONFIGS
	};

	enum SpinArray
	{
	  POSITION,
	  VELOCITY,
	};

	float update(float deltaTime, uint iterations);
	void reset(SpinConfig config);
	void resetPhase();

	void setGradient(float3 x) { m_gradient = x; }
	void getGradient(float3 &x) { x = m_gradient;}

	float* getArray();//SpinArray array);
	void   setArray();

	int getNumSpins() const { return m_numSpins; }
	float getSpinRadius() const { return m_spinRadius; }

	unsigned int getCurrentReadBuffer() const { return m_posVbo[m_currentPosRead]; }
	unsigned int getColorBuffer() const { return m_colorVBO; }

	void dumpGrid();
	void dumpSpins(uint start, uint count);
	double getMrSignal();

	void setAdc(float extraAdc, float intraAdc, float myelinAdc) { m_extraAdc = extraAdc; m_intraAdc = intraAdc; m_myelinAdc = myelinAdc;}
	void setT2(float extraT2, float intraT2, float myelinT2) { m_T2outside = extraT2; m_T2fiber = intraT2; m_T2myelin = myelinT2;}
	void setPermeability(float x) { m_permeability = x; }

	//float m_T2fiber;
	//float m_T2myelin;
	//float m_T2outside;

	int getNumFibers() { return m_numFibers; }

	float *getFiberPos(int i) { return &(m_fiberPos[i*4]); }

	float getOCERadius(int i) { return m_fiberPos[i*4+3]; }

	unsigned int getNumCubes() { return m_numCubes; }
	float getCubeLength() { return m_cubeLength; }
	
	uint getNumTimeSteps() { return m_totalNumTimeSteps; }

	void colorSphere(float *pos, float radius, float *color);
	void setColorFromSpin();
	void printOutPositions();
	bool addEntities();
	void printOutCells();
	float3 calcCubePos(float3 point);
	int calcCubeHash(float3 cubePos);

	//float *cellLevelTubeList;   

	void dumpPositionsIntoFile(FILE *fptr){
	    for(uint i=0;i<m_numSpins*3;i+=3)
	    fprintf(fptr,"%g %g %g ",m_hPos[i],m_hPos[i+1],m_hPos[i+2]);
	    fprintf(fptr,"\n");
	}
	void assignSeeds();
	//int findNearestFiber(int index);
	uint2 findNearestFiber(int index);

	void cpuIntegrateSystem(float deltaTime,float intraAdc,float extraAdc,float myelinAdc,float3 magneticGradient,float phaseConstant);

	void cpuIntegrate(int index,float3 magneticGradient,float phaseConstant,float intraStdDev,float extraStdDev, float myelinStdDev, float deltaTime, float inRadPropSq); 
	
protected: 
	// methods
	SpinSystem(){}
	uint createVBO(uint size);

	void _finalize();
	void initGrid(uint *size, float spacing, float jitter, uint numSpins);

	// data
	bool m_useGpu;
	bool m_useVbo;
	bool m_bInitialized;
	uint m_numSpins;
	uint m_totalNumTimeSteps;

	float m_T2fiber;
	float m_T2myelin;
	float m_T2outside;

	float3 m_gradient;
	// CPU data
	uint m_nParams;									// Change 4/2: Added variable

	float* m_hPos;
	float* m_hPartParams;								// Change 4/2: Added variable
	uint*  m_hSeed;

	uint*  m_hSpinHash;
	uint*  m_hCubeCounter;
	uint*  m_hCubes;

	// GPU data
	float m_maxFiberRadius;

	float* m_dPos[2];
	uint* m_dSeed[2];

	uint*  m_dSpinHash[2];

	float* m_dSortedPos;
	uint* m_dSortedSeed;

	float* m_dFiberPos;

    // A CUDA timer
    uint m_timer;

	// uniform cubes data
	uint*  m_dCubeCounter; // counts number of entries per cube
	uint*  m_dCubes;    // contains indices of up to "m_maxSpinsPerCube" spins per cube

	uint m_posVbo[2];
	uint m_colorVBO;

	uint m_currentPosRead, m_currentSeedRead;
	uint m_currentPosWrite, m_currentSeedWrite;

	// params
	uint m_numCubes;
	uint m_totalNumCubes;
	float m_cubeLength;
	uint m_maxFibersPerCube;
	
	// To do: allow each fiber to have a separate myelin radius.
	float m_innerRadiusProportion;

	float m_spinRadius;
	float m_damping;
	float m_extraAdc;
	float m_intraAdc;
	float m_myelinAdc;
	float m_intraAdcScale;
	float m_spaceScale;
	float m_permeability;

	float m_gyromagneticRatio;

	uint m_numFibers;
	float *m_fiberPos;

	void boxMuller(float& u1, float& u2);
	uint rand31pmc(uint &seed);
	uint myRand(uint seed[]);
	void myRandn(uint seed[], float& n1, float& n2, float& n3, float& u);

	float distToCenterSq(float3 partPos, float3 fiberPos);
	int signum(float x);
	float3 reflectedPos(float3 startPos, float3 targetPos, float3 fiberPos, float radiusSq, uint imo, bool outerMembrane, float inRadPropSq);
};

#endif // __BODYSYSTEMCUDA_H__
