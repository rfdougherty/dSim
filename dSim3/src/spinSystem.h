#ifndef __BODYSYSTEMCUDA_H__
#define __BODYSYSTEMCUDA_H__
#include <fstream>
#define DEBUG_GRID 0
#define DO_TIMING 0

#include <math.h>
#include "RTree.h"
#include "dSimDataTypes.h"

typedef unsigned int uint;

typedef struct _int3
{
	int x;
	int y;
	int z;
	
}int3;

typedef struct _uint3
{
	uint x;
	uint y;
	uint z;
	
}uint3;

typedef struct _float3
{
	float x;
	float y;
	float z;
}float3;

typedef struct _collResult
{
	//bool collision;
	uint collisionType;	
	float3 collPoint;
	uint collIndex;
	float collDistSq;
}collResult;


// Some simple inline vector ops for float3's, to mimick those found in cudautil_math.h.
inline float3 make_float3(float x, float y, float z){
	float3 p;
	p.x=x; p.y=y; p.z=z;
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






class SpinSystem {
	public:
		SpinSystem(int numSpins, bool useGpu, float spaceScale, float gyroMagneticRatio, bool useVbo, uint triSearchMethod, uint reflectionType, 
			float extraAdc, float myelinAdc, float intraAdc, float permeability, float extraT2, float intraT2, float myelinT2, float deltaTime, float startBoxSize);
		~SpinSystem();
		bool build();
		//bool initFibers(FILE *fiberFilePtr);
		bool initFibers(char * fiberFile);
		int getNumSpins() const { return m_numSpins; }
		unsigned int getCurrentReadBuffer() const { return m_posVbo[m_currentPosRead]; }
		void resetSpins();
		void assignSeeds();
		float* getArray();
		//float* getSpinArray();								// Test
		spinData* getSpinArray();
		void setArray();
		void setSpinArray();								// Test
		void findSpinsInFibers();
		float updateSpins(float deltaTime, uint iterations);
		void setGradient(float3 x){m_gradient = x;}
		double getMrSignal();
		void getMrSignal(double mrSignal[]);
		void constructAllRTrees();
		

		void cpuIntegrateSystem(float deltaTime, /*float intraAdc, float extraAdc, float myelinAdc, float gliaAdc,*/ float3 magneticGradient, float phaseConstant);
		void cpuIntegrate(int spinIndex, float3 magneticGradient, float phaseConstant,/* float intraStdDev, float extraStdDev, float myelinStdDev, float gliaStdDev,*/ float deltaTime/*, float radPrSq*/);
		unsigned int getColorBuffer(){ return m_colorVBO; }
		void setColorFromSignal();

		//float3 calcCubePos(float3 point);
		//int calcCubeHash(float3 cubePos);
		uint3 calcCubePos(float3 point);
		uint calcCubeHash(uint3 cubePos);
		void populateCubes();
		uint getNumTriInMembraneType(uint membraneType);
		float3 getTriPoint(uint nTri,uint nPoint);
		uint getNumFibers();
		uint getNumCompartments();
		uint getNumTriInFiberMembrane(uint membraneType, uint fiberNr);
		uint getTriInFiberArray(uint layerIndex, uint fiberIndex, uint triIndex);



	protected:
		// Methods
		SpinSystem(){}
		uint createVBO(uint size);
		
		void _finalize();
		void boxMuller(float& u1, float& u2);
		uint myRand(uint seed[]);
		void myRandn(uint seed[], float& n1, float& n2, float& n3, float& u);
		float3 collDetect(float3 oPos, float3 pos, float u, uint detectMethod, uint8 &compartment, uint16 &fiberInside);
		float3 collDetectRTree(float3 oPos, float3 pos, float u, uint8 &compartment, uint16 &fiberInside);
		float3 collDetectRectGrid(float3 pPos, float3 pos, float u, uint8 &compartment, uint16 &fiberInside);
		collResult cubeCollDetect(float3 oPos, float3 pos, uint cubeIndex, uint excludedTriangle);
		collResult triCollDetect(float3 oPos, float3 pos, uint triIndex);
		//float3 reflectPos(float3 startPos, float3 targetPos, float3 interPos, uint interIndex, uint collisionType, uint reflectType);
		float3 reflectPos(float3 startPos, float3 targetPos, float3 collPos, uint collTriIndex, uint collisionType);
		uint calcCubeIntersect(float3 startPos, float3 endPos, uint cubeArray[]);
		uint SearchTreeArray(float* rect, float* RTreeArray, uint* interSectArray);
		//void constructRTree(uint membranes[], uint numMembranes);
		//void createRTree(RTree<uint, float, 3, float> &tree, uint membranes[], uint numMembranes);
		void createRTree(RTree<uint, float, 3, float> &tree, uint *membranesInTree, uint *fibersInTree);
		//RTree<uint, float, 3, float> testTreeCreator();
		float* createRTreeArray(RTree<uint, float, 3, float> &tree, uint &treeArraySize);
		//float* testArrayConstructor();
		//float* testArrayConstructor2(RTree<uint, float, 3, float> &tree);
		bool pointInCube(float3 p, float3 rectMin, float3 rectMax);
		bool pointIn2DTriangle(float px,float py,float t1x,float t1y,float t2x,float t2y,float t3x,float t3y);
		bool lineIntersect(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4);
		bool rayTriangleIntersect(float3 rayPoint1, float3 rayPoint2, float3 triPoint1, float3 triPoint2, float3 triPoint3);

		// Data
		float m_deltaTime;
		uint m_reflectionType;
		uint m_triSearchMethod;
		bool m_useDisplay;
		uint m_numSpins;
		uint m_nPosValues;
		uint m_nSeedValues;
		uint m_nSpinValues;
		uint m_nAxonFibers;
		uint m_nMyelinFibers;
		uint m_nGliaFibers;
		uint m_nVertices_temp;
		uint m_nTriangles_temp;
		uint m_nMembraneTypes_temp;
		uint m_nFibers_temp;
		uint m_nMaxTrianglesPerMembrane_temp;
		//float m_extraAdc;
		//float m_myelinAdc;
		//float m_intraAdc;
		//float m_gliaAdc;
		//float m_extraT2;
		//float m_myelinT2;
		//float m_intraT2;
		float m_spinRadius;
		float m_permeability;
		float m_spaceScale;
		float m_gyroMagneticRatio;
		float m_innerRadiusProportion;
		bool m_useGpu;
		float m_startBoxSize;
		float3 m_gradient;
		uint m_numCubes;
		uint m_totalNumCubes;
		float m_cubeLength;
		uint m_maxTrianglesPerCube;
		RTree<uint, float, 3, float> totalTree;
		RTree<uint, float, 3, float>* m_treeGroup;
		uint m_numCompartments;

		float* m_hPos;
		uint* m_hSeed;
		//spinData* m_hSpins;
		//float* m_hSpins;
		spinData *m_hSpins;

		float* m_hRTreeArray;
		uint* m_hTrianglesInCubes;
		uint* m_hCubeCounter;
		//float m_triangles [300][21];		// Am phasing this out
		uint m_nVertices;
		uint m_nTriangles;
		uint* m_nTrianglesInMembraneType;
		uint m_maxTrianglesOnSurface;
		uint m_nFibers;
		uint m_nMembraneTypes;
		float* m_vertices;
		uint* m_trgls;
		uint* m_fibers;
		float* m_triangleHelpers;
		uint* m_triCounter;
		float* m_hStdDevs;
		float* m_hT2Values;
		uint* m_hTriInfo;
		float m_xmax, m_ymax, m_zmax;

		uint *m_hTestArray;			// Test
		uint *m_dTestArray;			// Test

		uint m_posVbo[2];
		uint m_colorVBO;
		uint m_currentPosRead;
		uint m_currentSeedRead;
		uint m_currentPosWrite;
		uint m_currentSeedWrite;
		uint m_currentSpinRead;			// Test
		uint m_currentSpinWrite;		// Test

		// GPU data
		float* m_dPos[2];
		uint* m_dSeed[2];
		//float* m_dSpins[2];
		spinData* m_dSpins[2];			// Test
		uint* m_dCubeCounter;			// Test
		uint* m_dTrianglesInCubes;		// Test
		//uint* m_dTrgls;			// Test
		uint *m_dTriInfo;
		float* m_dVertices;			// Test
		float* m_dTriangleHelpers;		// Test
		float* m_dRTreeArray;			// Test
		uint* m_dTreeIndexArray;
		float* m_dStdDevs;			// Test
		float* m_dT2Values;			// Test
};

#endif // __BODYSYSTEMCUDA_H__
