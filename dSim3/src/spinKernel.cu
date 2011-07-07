///////////////////////////////////////////////////////////////////////////////////////
// File name:		spinKernel.cu
// Description:		Kernel for spin computations using GPU
///////////////////////////////////////////////////////////////////////////////////////

#ifndef _SPIN_KERNEL_H_
#define _SPIN_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#include "math_constants.h"
#include "options.h"
#include "dSimDataTypes.h"

#define PI 3.14159265358979f
#define TWOPI 6.28318530717959f


//////////////////////////////////////////////////////////////////////////////////
// Define texture arrays and constants, copied to device from host.
//////////////////////////////////////////////////////////////////////////////////
texture<uint,1,cudaReadModeElementType> texCubeCounter;
texture<uint,1,cudaReadModeElementType> texTrianglesInCubes;
//texture<uint,1,cudaReadModeElementType> texTrgls;
texture<float,1,cudaReadModeElementType> texVertices;
texture<float,1,cudaReadModeElementType> texTriangleHelpers;
texture<float,1,cudaReadModeElementType> texRTreeArray;
texture<uint,1,cudaReadModeElementType> texCombinedTreeIndex;
texture<uint,1,cudaReadModeElementType> texTriInfo;

__constant__ uint k_reflectionType;
__constant__ uint k_triSearchMethod;
__constant__ uint k_numCubes;
__constant__ uint k_totalNumCubes;
__constant__ uint k_maxTrianglesPerCube;
__constant__ float k_cubeLength;
__constant__ uint k_nFibers; 
__constant__ uint k_nCompartments;
__constant__ float k_permeability;
__constant__ float k_deltaTime;
__constant__ float *k_T2Values;
__constant__ float *k_stdDevs;

typedef unsigned int uint;

/////////////////////////////////////////////////////////////////////////////////////
// The structure collResult will be used to store outcomes from checks of whether
// collision occurs between a ray and a triangle.
/////////////////////////////////////////////////////////////////////////////////////
typedef struct _collResult
{
	uint collisionType;			// 0 if no collision, 1 if collision within triangle, 2 if collision with triangle edge, 3 if collision with triangle vertex
	float3 collPoint;			// Point of collision with triangle
	uint collIndex;				// Index of collision triangle
	float collDistSq;			// Distance squared from starting point to collision point
}collResult;


// Some simple vector ops for float3's (dot and length are defined in cudautil_math)
//#define dot(u,v)   ((u).x * (v).x + (u).y * (v).y + (u).z * (v).z)
//#define length(v)    sqrt(dot(v,v))  // norm (vector length)
#define d(u,v)	length(u-v)	// distance (norm of difference)


//////////////////////////////////////////////////////////////////////////
// Function name:	point_line_dist
// Description:		Returns the shortest distance from a point P to a 
//			line defined by two points (LP1 and LP2)
//////////////////////////////////////////////////////////////////////////
__device__ float point_line_dist(float3 P, float3 LP1, float3 LP2){
	float3 v = LP2-LP1;
	float b = dot(P-LP1,v)/dot(v,v);
	return d(P,LP1+b*v);
}


///////////////////////////////////////////////////////////////////////////
// Function name:	point_seg_dist
// Description:		Returns the shortest distance from a point P to a 
//			line segment defined by two points (SP1 and SP2)
///////////////////////////////////////////////////////////////////////////
__device__ float point_seg_dist(float3 P, float3 SP1, float3 SP2){
	float3 v = SP2-SP1;
	float c1 = dot(P-SP1,v);
	if (c1<=0) return d(P,SP1);
	float c2 = dot(v,v);
	if (c2<=c1) return d(P,SP2);
	float3 Pb = SP1 + c1/c2*v;
	return d(P,Pb);
}


//////////////////////////////////////////////////////////////////////////////
// Function name:	boxMuller
// Description:		Generates a pair of independent standard normally
//			distributed random numbers from a pair of
//			uniformly distributed random numbers, using the basic form
//			of the Box-Muller transform 
//			(see http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)
//////////////////////////////////////////////////////////////////////////////
__device__ void boxMuller(float& u1, float& u2){
	float r = sqrtf(-2.0f * __logf(u1));
	float phi = TWOPI * u2;
	u1 = r * __cosf(phi);
	u2 = r * __sinf(phi);
}


//////////////////////////////////////////////////////////////////////////////
// Function name:	myRand
// Description:		Simple multiply-with-carry PRNG that uses two seeds 
//			(seed[0] and seed[1]) (Algorithm from George Marsaglia: 
//			http://en.wikipedia.org/wiki/George_Marsaglia)
//////////////////////////////////////////////////////////////////////////////
//__device__ uint myRand(uint seed[]){
//	seed[0] = 36969 * (seed[0] & 65535) + (seed[0] >> 16);
//	seed[1] = 18000 * (seed[1] & 65535) + (seed[1] >> 16);
//	return (seed[0] << 16) + seed[1];
//}
__device__ uint myRand(uint2 &seed){
	seed.x = 36969 * (seed.x & 65535) + (seed.x >> 16);
	seed.y = 18000 * (seed.y & 65535) + (seed.y >> 16);
	return (seed.x << 16) + seed.y;
}


/////////////////////////////////////////////////////////////////////////////
// Function name:	myRandf
// Description:		Returns a random float r in the range 0<=r<=1
/////////////////////////////////////////////////////////////////////////////
//__device__ float myRandf(uint seed[]){
//	return ((float)myRand(seed) / 4294967295.0f);
//}


/////////////////////////////////////////////////////////////////////////////
// Function name:	myRandDir
// Description:		Return a vector with a specified magnitude (adc) and 
//			a random direction
/////////////////////////////////////////////////////////////////////////////
//__device__ void myRandDir(uint seed[], float adc, float3& vec){
//	// Azimuth and elevation are on the interval [0,2*pi]
//	// (2*pi)/4294967294.0 = 1.4629181e-09f
//	float az = (float)myRand(seed) * 1.4629181e-09f;
//	float el = (float)myRand(seed) * 1.4629181e-09f;
//	vec.z = adc * __sinf(el);
//	float rcosel = adc * __cosf(el);
//	vec.x = rcosel * __cosf(az);
//	vec.y = rcosel * __sinf(az);
//	return;
//}


//////////////////////////////////////////////////////////////////////////////
// Function name:	myRandn
// Description:		Returns three normally distributed random numbers 
//			and one uniformly distributed random number.
//////////////////////////////////////////////////////////////////////////////
/*__device__ void myRandn(uint seed[], float& n1, float& n2, float& n3, float& u){
	// We want random numbers in the range (0,1], i.e. 0<n<=1
	n1 = ((float)myRand(seed) + 1.0f) / 4294967296.0f;
	n2 = ((float)myRand(seed) + 1.0f) / 4294967296.0f;
	n3 = ((float)myRand(seed) + 1.0f) / 4294967296.0f;
	u = ((float)myRand(seed) + 1.0f) / 4294967296.0f;
	// Note that ULONG_MAX=4294967295
	float n4 = u;
	boxMuller(n1,n2);
	boxMuller(n3,n4);
	return;
}*/
__device__ void myRandn(uint2 &seed, float& n1, float& n2, float& n3, float& u){
	// We want random numbers in the range (0,1], i.e. 0<n<=1
	n1 = ((float)myRand(seed) + 1.0f) / 4294967296.0f;
	n2 = ((float)myRand(seed) + 1.0f) / 4294967296.0f;
	n3 = ((float)myRand(seed) + 1.0f) / 4294967296.0f;
	u = ((float)myRand(seed) + 1.0f) / 4294967296.0f;
	// Note that ULONG_MAX=4294967295
	float n4 = u;
	boxMuller(n1,n2);
	boxMuller(n3,n4);
	return;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function name:	calcCubePosGPU										// Rename later to calcCubePos(...)
// Description: 	Function calculates the cube cell to which the given position belongs in uniform cube.
//			Converts a position coordinate (ranging from (-1,-1,-1) to (1,1,1) to a cube
//			coordinate (ranging from (0,0,0) to (m_numCubes-1, m_numCubes-1, m_numCubes-1)).
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ uint3 calcCubePosGPU(float3 p){
	uint3 cubePos;
	cubePos.x = floor((p.x + 1.0f) / k_cubeLength);
	cubePos.y = floor((p.y + 1.0f) / k_cubeLength);
	cubePos.z = floor((p.z + 1.0f) / k_cubeLength);

	cubePos.x = max(0, min(cubePos.x, k_numCubes-1));
	cubePos.y = max(0, min(cubePos.y, k_numCubes-1));
	cubePos.z = max(0, min(cubePos.z, k_numCubes-1));

	return cubePos;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function name:	calcCubeHashGPU										// Rename later to calcCubeHash(...)
// Description:		Calculate address in cube from position (clamping to edges)
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ uint calcCubeHashGPU(uint3 cubePos){							
	return cubePos.z * k_numCubes * k_numCubes + cubePos.y * k_numCubes + cubePos.x;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function name:	reflectPos
// Description:		Given a particle that tries to travel from startPos to targetPos, but collides with triangle
//			number collTriIndex at collPos, we calculate the position which the particle gets reflected to.
//				This applies if reflectionType==1. If reflectionType==0, we do a simplified reflection,
//			where the particle just gets reflected to its original position. This is also done if we hit
//			a triangle edge or a triangle vertex (which gives collisionType==2 or collisionTYpe==3).
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float3 reflectPos(float3 startPos, float3 targetPos, float3 collPos, uint collTriIndex, uint collisionType){

	float3 reflectedPos;

	if ((k_reflectionType==0)|(collisionType>1)){			// We simply reflect back to the starting point
			reflectedPos = startPos;
	} else {				// We reflect the target point through the triangle - see http://en.wikipedia.org/wiki/Transformation_matrix
			float3 sPosShifted = targetPos-collPos;
			float3 normalVec;
			normalVec = make_float3(tex1Dfetch(texTriangleHelpers,collTriIndex*12+0),tex1Dfetch(texTriangleHelpers,collTriIndex*12+1),tex1Dfetch(texTriangleHelpers,collTriIndex*12+2));
			reflectedPos.x = (1-2*normalVec.x*normalVec.x)*sPosShifted.x - 2*normalVec.x*normalVec.y*sPosShifted.y - 2*normalVec.x*normalVec.z*sPosShifted.z + collPos.x;
			reflectedPos.y = -2*normalVec.x*normalVec.y*sPosShifted.x + (1-2*normalVec.y*normalVec.y)*sPosShifted.y - 2*normalVec.y*normalVec.z*sPosShifted.z + collPos.y;
			reflectedPos.z = -2*normalVec.x*normalVec.z*sPosShifted.x - 2*normalVec.y*normalVec.z*sPosShifted.y + (1-2*normalVec.z*normalVec.z)*sPosShifted.z + collPos.z;
	}

	return reflectedPos;
}


//////////////////////////////////////////////////////////////////////////////////////////////
// Function name:	triCollDetect
// Description:		Find whether the path from oPos to pos intersects triangle no. triIndex.
// 			Returns the collision result, which consists of 
// 				result.collPoint = the collision/intersection point between 
//							the ray and the triangle.
// 				result.collIndex = the index of the collision triangle if 
//							collision occurs
// 				result.collisionType = 0 if no collision, 1 within triangle boundaries,
//							2 if collision with triangle edge, 3 if 
//							collision with triangle vertex
// 				result.collDistSq = the distance (squared) from oPos to 
//							the collision point.
//////////////////////////////////////////////////////////////////////////////////////////////
__device__ collResult triCollDetect(float3 oPos, float3 pos, uint triIndex){

	uint firstPointIndex;
	float uv, uu, vv, wu, wv, r, s, t, stDen;
	float3 triP1, d, w, n, u, v, collPoint;
	collResult result;
	result.collisionType = 0;
	
	// firstPointIndex is the index of the "first" point in the triangle
	firstPointIndex = tex1Dfetch(texTriInfo, triIndex*3+2);
	// triP1 holds the coordinates of the first point
	triP1 = make_float3(tex1Dfetch(texVertices,firstPointIndex*3+0),tex1Dfetch(texVertices,firstPointIndex*3+1),tex1Dfetch(texVertices,firstPointIndex*3+2));
	// n: normal to the triangle. u: vector from first point to second point. v: vector from first point to third point. uv, uu, vv: dot products.
	n = make_float3(tex1Dfetch(texTriangleHelpers,triIndex*12+0),tex1Dfetch(texTriangleHelpers,triIndex*12+1),tex1Dfetch(texTriangleHelpers,triIndex*12+2));
	u = make_float3(tex1Dfetch(texTriangleHelpers,triIndex*12+3),tex1Dfetch(texTriangleHelpers,triIndex*12+4),tex1Dfetch(texTriangleHelpers,triIndex*12+5));
	v = make_float3(tex1Dfetch(texTriangleHelpers,triIndex*12+6),tex1Dfetch(texTriangleHelpers,triIndex*12+7),tex1Dfetch(texTriangleHelpers,triIndex*12+8));

	uv = tex1Dfetch(texTriangleHelpers,triIndex*12+9);
	uu = tex1Dfetch(texTriangleHelpers,triIndex*12+10);
	vv = tex1Dfetch(texTriangleHelpers,triIndex*12+11);

	// First find whether the path intersects the plane defined by triangle i. See method at http://softsurfer.com/Archive/algorithm_0105/algorithm_0105.htm
	r = dot(n,triP1-oPos)/dot(n,pos-oPos);

	if ((0<r)&(r<1)){
	// Then find if the path intersects the triangle itself. See method at http://softsurfer.com/Archive/algorithm_0105/algorithm_0105.htm
		d = r*(pos-oPos);
		collPoint = oPos + d;
		w = collPoint-triP1;

		wu = dot(w,u);
		wv = dot(w,v);

		stDen = uv*uv-uu*vv;
		s = (uv*wv-vv*wu)/stDen;
		t = (uv*wu-uu*wv)/stDen;

		if ( (s>=0)&(t>=0)&(s+t<=1) ){	// We have a collision with the triangle

			result.collDistSq = dot(d,d);
			result.collIndex = triIndex;
			result.collPoint = collPoint;
			result.collisionType = 1;

			if ( (s==0)|(t==0)|(s+t==1) ){						// The collision point is on a triangle edge
				result.collisionType = 2;
						
				if ( ((s==0)&(t==0))|((s==0)&(t==1))|((s==1)&(t==0)) ){		// The collision point is on a triangle vertex
					result.collisionType = 3;
				}							
			}
		}
	}
	return result;
}


/////////////////////////////////////////////////////////////////////////////////////////
// Function name:	SearchRTreeArray
// Description:		Find the leaf rectangles in the R-Tree which intersect the rectangle
//			rect (=[x_min,y_min,z_min,x_max,y_max,z_max]). Normally, rect will
//			be a bounding rectangle for a particle path and the leaf rectangles
//			of the R-Tree will be bounding rectangles for fiber triangles.
//			When the rectangles intersect, that means the particle might collide
//			with the triangle. The indices of such triangles are written into
//			intersectArray (to be further checked for actual collisions), and
//			the number of intersecting rectangles is returned in the output
//			foundCount.
/////////////////////////////////////////////////////////////////////////////////////////
__device__ uint SearchRTreeArray(float* rect, uint* interSectArray, uint8 &compartment, uint16 &fiberInside){
	uint foundCount = 0;
	uint stack[100];		// Maximum necessary stack size should be 1+7*(treeHeight) = 1+7*(n_levels-1). 100 should suffice for n_levels <= 15 - very big tree	
	int stackIndex = 0;
	
	//printf("k_nFibers: %u\n", k_nFibers);
	//printf("k_nCompartments: %u\n", k_nCompartments);
	//uint k_nFibers = 17, k_nCompartments = 3;
	//stack[stackIndex] = 0;
	if (compartment != 0){										// We push the location of the root node onto the stack
		stack[stackIndex] = tex1Dfetch(texCombinedTreeIndex,fiberInside*(k_nCompartments-1)+compartment);	// = 0 for "first" tree, i.e. tree corresponding to innermost compartment
	} else{
		stack[stackIndex] = tex1Dfetch(texCombinedTreeIndex,0);
		//printf("k_nFibers: %u\n", k_nFibers);
		//printf("k_nCompartments: %u\n", k_nCompartments);
		//printf("StackIndex: %u\n", stackIndex);
		//printf("Stack for compartment %u: %i\n", compartment, stack[stackIndex]);
	}
	//printf("(in spinKernel.cu::SearchRTreeArray): rect: [%g,%g,%g,%g,%g,%g]\n", rect[0],rect[1],rect[2],rect[3],rect[4],rect[5]);
	//printf("(in spinKernel.cu::SearchRTreeArray): stack[%i]: %u\n", stackIndex, stack[stackIndex]);
	stackIndex++;

	uint currentNodeIndex;
	


	while (stackIndex > 0){					// Stop when we've emptied the stack
		stackIndex--;					// Pop the top node off the stack
		currentNodeIndex = stack[stackIndex];
		//printf("(in spinKernel.cu::SearchRTreeArray): currentNodeIndex: %u\n", currentNodeIndex);

		for (int m=tex1Dfetch(texRTreeArray,currentNodeIndex+1)-1; m>=0; m--){
			uint currentBranchIndex = currentNodeIndex+2 + m*7;
			//printf("(in spinKernel.cu::SearchRTreeArray): m: %u\n", m);
			//printf("(in spinKernel.cu::SearchRTreeArray): currentBranchIndex: %u\n", currentBranchIndex);

			//See if the branch rectangle overlaps with the input rectangle
			if (!(  tex1Dfetch(texRTreeArray,currentBranchIndex+1) > rect[3] ||		// branchRect.x_min > rect.x_max
				tex1Dfetch(texRTreeArray,currentBranchIndex+2) > rect[4] ||		// branchRect.y_min > rect.y_max
				tex1Dfetch(texRTreeArray,currentBranchIndex+3) > rect[5] ||		// branchRect.z_min > rect.z_max
				rect[0] > tex1Dfetch(texRTreeArray,currentBranchIndex+4) ||		// rect.x_min > branchRect.x_max
				rect[1] > tex1Dfetch(texRTreeArray,currentBranchIndex+5) ||		// rect.y_min > branchRect.y_max
				rect[2] > tex1Dfetch(texRTreeArray,currentBranchIndex+6) ))		// rect.z_min > branchRect.z_max
			{	
				if (tex1Dfetch(texRTreeArray,currentNodeIndex) > 0){		// We are at an internal node - push the node pointed to in the branch onto the stack
					stack[stackIndex] = tex1Dfetch(texRTreeArray,currentBranchIndex);
					stackIndex++;
					//printf("(in spinKernel.cu::SearchRTreeArray): stackIndex: %i\n", stackIndex);
				} else {
					interSectArray[foundCount] = tex1Dfetch(texRTreeArray,currentBranchIndex); // We are at a leaf - store corresponding triangle index
					foundCount++;
					//printf("(in spinKernel.cu::SearchRTreeArray): Tree rectangle: [%g,%g,%g,%g,%g,%g]\n", tex1Dfetch(texRTreeArray,currentBranchIndex+1), tex1Dfetch(texRTreeArray,currentBranchIndex+2),
					//tex1Dfetch(texRTreeArray,currentBranchIndex+3), tex1Dfetch(texRTreeArray,currentBranchIndex+4), tex1Dfetch(texRTreeArray,currentBranchIndex+5),
					//tex1Dfetch(texRTreeArray,currentBranchIndex+6));
				}
			}
		}
	}
	return foundCount;
}


//////////////////////////////////////////////////////////////////////////////////////////
// Function name:	collDetectRTree
// Description:		See whether a particle trying to go from startPos to targetPos
//			collides with any triangle in the mesh, using the R-Tree. Return
//			the final position of the particle.
//////////////////////////////////////////////////////////////////////////////////////////
__device__ float3 collDetectRTree(float3 startPos, float3 targetPos, float u, uint8 &compartment, uint16 &fiberInside){
	
	float3 endPos = targetPos;
	uint hitArray[1200];				// Hitarray will store the indices of the triangles that the particle possible collides with - we are assuming no more than 100
	float spinRectangle[6];
	collResult result, tempResult;
	//float minCollDistSq;
	result.collDistSq = 400000000;			// Some really large number, will use this to store the smallest distance to a collision point
	result.collisionType = 1;
	result.collIndex = UINT_MAX;
	uint excludedTriangle = UINT_MAX;
	float u_max = 1, u_min = 0;
	//uint k = 0;
	//uint p = 0;

	//printf("Compartment: %i\n", compartment);

	while (result.collisionType>0){			// If we have detected a collision, we repeat the collision detection for the new, reflected path
		//minCollDistSq = 400000000;
		//printf("p: %u\n", p);
		//p++;
		result.collisionType = 0;		// First assume that the particle path does not experience any collisions

		// Define a rectangle that bounds the particle path from corner to corner
		// Finding minx, miny, minz
		spinRectangle[0] = startPos.x; if (targetPos.x < spinRectangle[0]){spinRectangle[0] = targetPos.x;}
		spinRectangle[1] = startPos.y; if (targetPos.y < spinRectangle[1]){spinRectangle[1] = targetPos.y;}
		spinRectangle[2] = startPos.z; if (targetPos.z < spinRectangle[2]){spinRectangle[2] = targetPos.z;}
	
		// Finding maxx, maxy, maxz
		spinRectangle[3] = startPos.x; if (targetPos.x > spinRectangle[3]){spinRectangle[3] = targetPos.x;}
		spinRectangle[4] = startPos.y; if (targetPos.y > spinRectangle[4]){spinRectangle[4] = targetPos.y;}
		spinRectangle[5] = startPos.z; if (targetPos.z > spinRectangle[5]){spinRectangle[5] = targetPos.z;}
	
		// Find the triangles whose bounding rectangles intersect spinRectangle. They are written to hitArray and their number is nHits.
		int nHits = SearchRTreeArray(spinRectangle, hitArray, compartment, fiberInside);
		//int nHits = 0;
		
		//printf("(in spinKernel.cu::collDetectRTree): nHits: %i\n", nHits);
		//printf("(in spinKernel.cu::collDetectRTree): Startpos: [%g,%g,%g]\n", startPos.x, startPos.y, startPos.z);
		//printf("(in spinKernel.cu::collDetectRTree): Targetpos: [%g,%g,%g]\n", targetPos.x, targetPos.y, targetPos.z);
		//printf("(in spinKernel.cu::collDetectRTree): Compartment: %i\n", compartment);
		//printf("(in spinKernel.cu::collDetectRTree): Fiber: %u\n", fiberInside);
		//printf("(in spinKernel.cu::collDetectRTree): Excluded triangle: %u\n", excludedTriangle);
		//printf("(in spinKernel.cu::collDetectRTree): result.collDistSq: %g\n", result.collDistSq);
	
		// Loop through the triangles in hitArray, see if we have collisions, store the closest collision point in the variable result.
		for (uint k=0; k<nHits; k++){
			uint triIndex = hitArray[k];
			//printf("(in spinKernel.cu::collDetectRTree): hitArray[%u]: %u\n", k, hitArray[k]);
			if (triIndex != excludedTriangle){
				tempResult = triCollDetect(startPos, targetPos, triIndex);
				//if ((tempResult.collisionType>0) & (tempResult.collDistSq < result.collDistSq)){
				if ((tempResult.collisionType>0) & (tempResult.collDistSq < result.collDistSq)){
					result = tempResult;
					//minCollDistSq = tempResult.collDistSq;
				}
			}
		}
		
	
		// If we have a collision, then we find the resulting point which the particle gets reflected to.
		if (result.collisionType>0){
			//printf("*\n");
			//printf("(in spinKernel.cu::collDetectRTree): Collision!\n");
			//printf("(in spinKernel.cu::collDetectRTree): startPos: [%g,%g,%g]\n", startPos.x,startPos.y,startPos.z);
			//printf("(in spinKernel.cu::collDetectRTree): targetPos: [%g,%g,%g]\n", targetPos.x,targetPos.y,targetPos.z);
			//printf("(in spinKernel.cu::collDetectRTree): Collision point: [%g,%g,%g]\n", result.collPoint.x, result.collPoint.y, result.collPoint.z);
			//printf("(in spinKernel.cu::collDetectRTree): Endpos (before assignment): [%g,%g,%g]\n", endPos.x, endPos.y, endPos.z);
			//printf("(in spinKernel.cu::collDetectRTree): Collision triangle index: %u\n", result.collIndex);
			//printf("(in spinKernel.cu::collDetectRTree): Collision fiber index: %u\n", tex1Dfetch(texTriInfo, result.collIndex*3+0));
			//printf("(in spinKernel.cu::collDetectRTree): Collision membrane index: %u\n", tex1Dfetch(texTriInfo, result.collIndex*3+1));
			//printf("(in spinKernel.cu::collDetectRTree): u: %g\n", u);
			//printf("(in spinKernel.cu::collDetectRTree): u_max: %g, u_min: %g, u_p: %g\n", u_max, u_min, u_max-(u_max-u_min)*k_permeability);

			// If u>u_max-(u_max-u_min)*k_permeability, then the particle permeates through the membrane and does not get reflected.
			// u is in the range (0,1].
			if (u<=u_max-(u_max-u_min)*k_permeability){		// The spin does not permeate the membrane
				endPos = reflectPos(startPos, targetPos, result.collPoint, result.collIndex, result.collisionType);
				u_max = u_max-(u_max-u_min)*k_permeability;
				//printf("(in spinKernel.cu::collDetectRTree): Particle bounces off membrane\n");
				//printf("(in spinKernel.cu::collDetectRTree): Endpos: [%g,%g,%g]\n", endPos.x, endPos.y, endPos.z);
				//reflectPos(startPos, targetPos, result.collPoint, result.collIndex, result.collisionType);
			} else{							// The spin permeates the membrane
				u_min = u_max-(u_max-u_min)*k_permeability;

				// Change the compartment (and fiber, if appropriate) assignment of the spin
				// uint membraneType = tex1Dfetch(texTriInfo, result.collIndex*3+1);
				if (compartment == 2){
					if (tex1Dfetch(texTriInfo, result.collIndex*3+1) == 0){		// We are going from compartment 2 through axon surface - new compartment is 1
						compartment = 1;
					} else {							// We are going from compartment 2 through myelin surface - new compartment is 0
						compartment = 0;
						fiberInside = UINT16_MAX;
					}
				} else if (compartment == 1){
					compartment = 2;						// We are going from compartment 1 through axon surface - new compartment is 2
				} else if (compartment == 3){
					compartment = 0;						// We are going from compartment 3 through glia surface - new compartment is 0
					fiberInside = UINT16_MAX;
				} else {
					fiberInside = tex1Dfetch(texTriInfo, result.collIndex*3+0);
					if (tex1Dfetch(texTriInfo, result.collIndex*3+1) == 1){		// We are going from compartment 0 through myelin surface - new compartment is 2
						compartment = 2;
					} else {							// We are going from compartment 0 through glia surface - new compartment is 3
						compartment = 3;
					}
				}
				
				//printf("(in spinKernel.cu::collDetectRTree): Particle permeates membrane\n");
				//printf("(in spinKernel.cu::collDetectRTree): Endpos: [%g,%g,%g]\n", endPos.x, endPos.y, endPos.z);
			}
		}

		// Redefine the start and end points for the reflected path, then repeat until no collision is detected.
		startPos = result.collPoint;
		targetPos = endPos;
		excludedTriangle = result.collIndex;					// Make sure we don't detect a collision with the triangle which the particle bounces from
		result.collDistSq = 400000000;
	}

	return endPos;
}


/////////////////////////////////////////////////////////////////////////////////////////////
// Function name:	cubeCollDetect
// Description:		Determine whether a particle traveling from oPos to pos experiences
//			a collision with any of the triangles in cube no. cubeIndex. Triangle
//			no. excludedTriangle is not checked - useful if the particle is bouncing
//			off that triangle.
/////////////////////////////////////////////////////////////////////////////////////////////
__device__ collResult cubeCollDetect(float3 oPos, float3 pos, uint cubeIndex, uint excludedTriangle, uint* trianglesInCubes, uint* cubeCounter){

	uint triIndex, k_max;
	collResult result, testCollision;
	result.collisionType = 0;
	result.collDistSq = 400000000;
	result.collIndex = UINT_MAX;

	// Loop through membrane types (layers) as appropriate
	//for (uint layerIndex = 0; layerIndex < 2; layerIndex++){					// Change later so not to loop through all membrane types
		//k_max = tex1Dfetch(texCubeCounter, layerIndex*k_totalNumCubes+cubeIndex);		// k_max: the number of triangles in cube cubeIndex on membrane type layerIndex
		//k_max = tex1Dfetch(texCubeCounter, cubeIndex);
		//cubeIndex = 1275;
		k_max = cubeCounter[cubeIndex];
		//printf("cubeCounter[%u]: %u\n", cubeIndex, k_max);
		for (uint k=0; k<k_max; k++){
			// triIndex is the number of the triangle being checked.
			//triIndex = tex1Dfetch(texTrianglesInCubes, (layerIndex*k_totalNumCubes+cubeIndex)*k_maxTrianglesPerCube+k);
//			triIndex = tex1Dfetch(texTrianglesInCubes, cubeIndex*k_maxTrianglesPerCube+k);
			triIndex = trianglesInCubes[cubeIndex*k_maxTrianglesPerCube+k];
			//printf("Checking triangle %u\n", triIndex);
			if (triIndex != excludedTriangle){
				testCollision = triCollDetect(oPos, pos, triIndex);

				if ( (testCollision.collisionType>0)&(testCollision.collDistSq<result.collDistSq) ){
					result = testCollision;
				}
			}
		}
		//triIndex = tex1Dfetch(texTrianglesInCubes, cubeIndex*k_maxTrianglesPerCube+k);
	//}

	return result;
}



///////////////////////////////////////////////////////////////////////////////////////////////
// Function name:	collDetectRectGrid
// Description:		Determine whether a particle trying to go from startPos to targetPos
//			collides with a triangle, using the method of a rectangular grid (as 
//			opposed	to an R-Tree)
///////////////////////////////////////////////////////////////////////////////////////////////
__device__ float3 collDetectRectGrid(float3 startPos, float3 targetPos, float u, uint8 compartment, uint16 fiberInside, uint* trianglesInCubes, uint* cubeCounter){

	float3 endPos = targetPos;
	collResult collCheck;
	collCheck.collisionType = 1;
	uint excludedTriangle = UINT_MAX, currCube;
	uint3 currCubexyz, startCubexyz, endCubexyz;
	int3 cubeIncrement;
	float u_max = 1.0f, u_min = 0.0f;

	while (collCheck.collisionType > 0){

		//startCube = calcCubeHashGPU(calcCubePosGPU(startPos, k_cubeLength), k_numCubes);		// The cube that the particle starts in
		//endCube = calcCubeHashGPU(calcCubePosGPU(targetPos, k_cubeLength), k_numCubes);			// The cube that the particle tries to end in
		
		startCubexyz = calcCubePosGPU(startPos);
		endCubexyz = calcCubePosGPU(targetPos);
		cubeIncrement.x = ( (endCubexyz.x>startCubexyz.x) - (endCubexyz.x<startCubexyz.x) );
		cubeIncrement.y = ( (endCubexyz.y>startCubexyz.y) - (endCubexyz.y<startCubexyz.y) );
		cubeIncrement.z = ( (endCubexyz.z>startCubexyz.z) - (endCubexyz.z<startCubexyz.z) );

		//printf("startCubexyz: [%u,%u,%u]\n", startCubexyz.x, startCubexyz.y, startCubexyz.z);
		//printf("endCubexyz: [%u,%u,%u]\n", endCubexyz.x, endCubexyz.y, endCubexyz.z);
		//printf("cubeIncrement: [%i,%i,%i]\n", cubeIncrement.x, cubeIncrement.y, cubeIncrement.z);

		collCheck.collisionType = 0;

		currCubexyz.x = startCubexyz.x;
		do {
			currCubexyz.y = startCubexyz.y;
			do {
				currCubexyz.z = startCubexyz.z;
				do {
					currCube = calcCubeHashGPU(currCubexyz);
					//printf("currCubexyz: [%u,%u,%u]\n", currCubexyz.x, currCubexyz.y, currCubexyz.z);
					collCheck = cubeCollDetect(startPos, targetPos, currCube, excludedTriangle, trianglesInCubes, cubeCounter);
					currCubexyz.z += cubeIncrement.z;
				} while ((currCubexyz.z != endCubexyz.z+cubeIncrement.z)&&(collCheck.collisionType == 0));
				currCubexyz.y += cubeIncrement.y;
			} while ((currCubexyz.y != endCubexyz.y+cubeIncrement.y)&&(collCheck.collisionType == 0));
			currCubexyz.x += cubeIncrement.x;
		} while ((currCubexyz.x != endCubexyz.x+cubeIncrement.x)&&(collCheck.collisionType == 0));



		/*while ((currCubexyz.x != endCubexyz.x+cubeIncrement.x)&&(collCheck.collisionType == 0)){
			while ((currCubexyz.y != endCubexyz.y+cubeIncrement.y)&&(collCheck.collisionType == 0)){
				while ((currCubexyz.z != endCubexyz.z+cubeIncrement.z)&&(collCheck.collisionType == 0)){
					currCubexyz.z += cubeIncrement.z;
					currCube = calcCubeHashGPU(currCubexyz);
					printf("currCubexyz: [%u,%u,%u]\n", currCubexyz.x, currCubexyz.y, currCubexyz.z);
					collCheck = cubeCollDetect(startPos, targetPos, currCube, excludedTriangle, trianglesInCubes, cubeCounter);
				}
				currCubexyz.y += cubeIncrement.y;
			}
			currCubexyz.x += cubeIncrement.x;
		}*/



		if (collCheck.collisionType > 0){

			//printf("(in collDetectRectGrid): Collision!\n");
			//printf("(in collDetectRectGrid): Startpos: [%g,%g,%g]\n", startPos.x, startPos.y, startPos.z);
			//printf("(in collDetectRectGrid): Targetpos: [%g,%g,%g]\n", targetPos.x, targetPos.y, targetPos.z);
			//printf("(in collDetectRectGrid): Collision pos: [%g,%g,%g]\n", collCheck.collPoint.x, collCheck.collPoint.y, collCheck.collPoint.z);
			//printf("(in collDetectRectGrid): Collision triangle: %u\n", collCheck.collIndex);
			//printf("(in collDetectRectGrid): Cube: %u\n", currCube);
			//printf("(in collDetectRectGrid): Compartment: %u\n", compartment);
			//printf("(in collDetectRectGrid): FiberInside: %u\n", fiberInside);
			
			if (u<=u_max-(u_max-u_min)*k_permeability){		// The spin does not permeate the membrane
				endPos = reflectPos(startPos, targetPos, collCheck.collPoint, collCheck.collIndex, collCheck.collisionType);
				u_max = u_max-(u_max-u_min)*k_permeability;
				//printf("(in spinKernel.cu::collDetectRTree): Particle bounces off membrane\n");
				//printf("(in spinKernel.cu::collDetectRTree): Endpos: [%g,%g,%g]\n", endPos.x, endPos.y, endPos.z);
				//reflectPos(startPos, targetPos, collCheck.collPoint, collCheck.collIndex, collCheck.collisionType);
			} else{							// The spin permeates the membrane
				u_min = u_max-(u_max-u_min)*k_permeability;

				// Change the compartment (and fiber, if appropriate) assignment of the spin
				// uint membraneType = tex1Dfetch(texTriInfo, collCheck.collIndex*3+1);
				if (compartment == 2){
					if (tex1Dfetch(texTriInfo, collCheck.collIndex*3+1) == 0){		// We are going from compartment 2 through axon surface - new compartment is 1
						compartment = 1;
					} else {							// We are going from compartment 2 through myelin surface - new compartment is 0
						compartment = 0;
						fiberInside = UINT16_MAX;
					}
				} else if (compartment == 1){
					compartment = 2;						// We are going from compartment 1 through axon surface - new compartment is 2
				} else if (compartment == 3){
					compartment = 0;						// We are going from compartment 3 through glia surface - new compartment is 0
					fiberInside = UINT16_MAX;
				} else {
					fiberInside = tex1Dfetch(texTriInfo, collCheck.collIndex*3+0);
					if (tex1Dfetch(texTriInfo, collCheck.collIndex*3+1) == 1){		// We are going from compartment 0 through myelin surface - new compartment is 2
						compartment = 2;
					} else {							// We are going from compartment 0 through glia surface - new compartment is 3
						compartment = 3;
					}
				}
			}
		}

		
		// Redefine the start and end points for the reflected path, then repeat until no collision is detected.
		startPos = collCheck.collPoint;
		targetPos = endPos;
		excludedTriangle = collCheck.collIndex;					// Make sure we don't detect a collision with the triangle which the particle bounces from
	}
	return endPos;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function name:	collDetect
// Description:		Determine whether a particle trying to travel from oPos to pos hits a triangle.
//			Use either the method of a rectangular grid or an R-Tree.
////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float3 collDetect(float3 oPos, float3 pos, float u, uint8 &compartment, uint16 &fiberInside, uint* trianglesInCubes, uint* cubeCounter){

	//if (k_triSearchMethod == 0){
		return collDetectRectGrid(oPos,pos,u,compartment,fiberInside,trianglesInCubes,cubeCounter);
	//} else {
	//	return collDetectRTree(oPos, pos, u, compartment, fiberInside);
	//}
	//return pos;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////
// Function name:	integrate
// Description:		"Main" function for GPU kernel computation, called from spinSystem.cu, invokes all
//			the functions above. Computes the spin movement and signal for each spin by
//			performing the below computation in parallel on multiple threads.
///////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void integrate(
				float3* oldPos,
				uint2* oldSeed,
				//float4* spinInfo,
				spinData* spinInfo,
				float deltaTime,
				float permeability,
				int numBodies,
				float gradX, float gradY, float gradZ,
				float phaseConstant,
				uint iterations, uint* trianglesInCubes, uint* cubeCounter){

	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index>=numBodies)
	return;

	float3 pos = oldPos[index];								// pos = particle position
	uint2 seed2 = oldSeed[index];								// seed4 = seed values (currently only using first 2 values)
	//float signalMagnitude = spinInfo[index].signalMagnitude;
	//float signalPhase = spinInfo[index].signalPhase;
	uint8 compartment = spinInfo[index].compartmentType;
	uint16 fiberInside = spinInfo[index].insideFiber;
	

/////////////////////////////////////////////////////////////////////////////////
// Now apply the brownian motion (free diffusion). We simulate brownian motion
// with a random walk where the x, y, and z componenets are drawn from a 
// normal distribution with mean 0 and standard deviation of sqrt(2*ADC*deltaTime).
// From wikipedia http://en.wikipedia.org/wiki/Random_walk:
//    In 3D, the variance corresponding to the Green's function of the diffusion equation is:
//       sigma^2 = 6*D*t
//    sigma^2 corresponds to the distribution associated to the vector R that links the two 
//    ends of the random walk, in 3D. The variance associated to each component Rx, Ry or Rz 
//    is only one third of this value (still in 3D).
// Thus, the standard deviation of each component is sqrt(2*ADC*deltaTime)
//////////////////////////////////////////////////////////////////////////////////

	//uint rseed[2];
	//rseed[0] = seed2.x;
	//rseed[1] = seed2.y;

	for (uint i=0; i<iterations; i++){

		// Take a random walk...
		// myRandn returns 3 PRNs from a normal distribution with mean 0 and SD of 1. 
		// So, we just need to scale these with the desired SD to get the displacements
		// for the random walk.
		// myRandn also returns a bonus uniformly distributed PRN as a side-effect of the 
		// Box-Muller transform used to generate normally distributed PRNs.
		float u;
		float3 brnMot;
		//myRandn(rseed, brnMot.y, brnMot.x, brnMot.z, u);
		myRandn(seed2, brnMot.y, brnMot.x, brnMot.z, u);
		float3 oPos = pos;						// Store a copy of the old position before we update it

		pos.x += brnMot.x * k_stdDevs[compartment];
		pos.y += brnMot.y * k_stdDevs[compartment];
		pos.z += brnMot.z * k_stdDevs[compartment];

		


		// Test
		if (index == 0){
			//printf("i = %u\n", i);
			//printf("index: %u\n", index);
			//printf("oPos: [%g,%g,%g]\n", oPos.x,oPos.y,oPos.z);
			//printf("pos: [%g,%g,%g]\n", pos.x,pos.y,pos.z);
			//printf("Compartment: %u\n", compartment);
			//printf("Fiberinside: %u\n", fiberInside);
			//printf("Signal magnitude: %g\n", signalMagnitude);
			//printf("Signal phase: %g\n", signalPhase);
			//printf("u (before assignment): %g\n", u);
			
			//printf("rseed after: [%u,%u]\n", rseed[0], rseed[1]);
			//printf("[%g,%g,%g,%g,%g,%g,%u,%u]\n", oPos.x, oPos.y, oPos.z, pos.x, pos.y, pos.z, compartment, fiberInside);

		
			//oPos.x = 0.0; oPos.y = 0.0; oPos.z = 0.01;		// oPos.x = 0.7; oPos.y = 0.0; oPos.z = 0.01;
			//pos.x = 0.1; pos.y = 0.2; pos.z = -0.01;		// pos.x = 0.632; pos.y = 0.067; pos.z = 0.01;
			//compartment = 1;
			//fiberInside = 0;
			//u = 0.9;
			//printf("u (after assignment): %g\n", u);
		}

		// Do a collision detection for the path the particle is trying to take
		pos = collDetect(oPos,pos,u,compartment,fiberInside,trianglesInCubes,cubeCounter);

		
		// Don't let the spin leave the volume
		if (pos.x > 1.0f)  { pos.x = 1.0f; /*signalMagnitude = 0.0;*/ }
		else if (pos.x < -1.0f) { pos.x = -1.0f; /*signalMagnitude = 0.0;*/ }
		if (pos.y > 1.0f)  { pos.y = 1.0f; /*signalMagnitude = 0.0;*/ }
		else if (pos.y < -1.0f) { pos.y = -1.0f; /*signalMagnitude = 0.0;*/ }
		if (pos.z > 1.0f)  { pos.z = 1.0f; /*signalMagnitude = 0.0;*/ }
		else if (pos.z < -1.0f) { pos.z = -1.0f; /*signalMagnitude = 0.0;*/ }

		// Update MR signal magnitude
		//signalMagnitude += -signalMagnitude/k_T2Values[compartment]*k_deltaTime;
		spinInfo[index].signalMagnitude += -spinInfo[index].signalMagnitude/k_T2Values[compartment]*k_deltaTime;
		
		// Update MR signal phase
		//signalPhase += (gradX * pos.x + gradY * pos.y + gradZ * pos.z) * phaseConstant;
		spinInfo[index].signalPhase += (gradX * pos.x + gradY * pos.y + gradZ * pos.z) * phaseConstant;

	}

	// Store new position
	//oldPos[index] = make_float4(pos, signalPhase);
	oldPos[index] = pos;

	// Store new seed values
	//oldSeed[index].x = rseed[0];
	//oldSeed[index].y = rseed[1];
	oldSeed[index].x = seed2.x;
	oldSeed[index].y = seed2.y;

	// Store new values of compartment and signal magnitude and phase
	//spinInfo[index].signalMagnitude = signalMagnitude;
	//spinInfo[index].signalPhase = signalPhase;
	spinInfo[index].compartmentType = compartment;
	spinInfo[index].insideFiber = fiberInside;
		
}

#endif
