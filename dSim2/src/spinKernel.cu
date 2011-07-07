/* 
 * MR Diffusion spin system code
 * ----------------------------
 *
 * This spin system class is implemented here.
 *
 * Copyright 2008 Bob Dougherty (bobd@stanford.edu) and 
 * Shyamsundar Gopalakrishnan (gshyam@stanford.edu).
 */
 
#ifndef _SPIN_KERNEL_H_
#define _SPIN_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#include "math_constants.h"
#include "spinKernel.cuh"
#include "options.h"

#define PI 3.14159265358979f
#define TWOPI 6.28318530717959f

texture<float4,1,cudaReadModeElementType> texFiberList;
texture<uint, 1,cudaReadModeElementType> texCubeList;

typedef unsigned int uint;

// Some simple vector ops for float3's (dot and length are defined in cudautil_math)
//#define dot(u,v)   ((u).x * (v).x + (u).y * (v).y + (u).z * (v).z)
//#define length(v)    sqrt(dot(v,v))  // norm (vector length)
#define d(u,v)     length(u-v)       // distance (norm of difference)

// Returns the shortest distance from a point P to a line defined
// by two points (LP1 and LP2)
__device__ float point_line_dist(float3 P, float3 LP1, float3 LP2){ 
   float3 v = LP2-LP1; 
   float b = dot(P-LP1, v) / dot(v, v);
   return d(P, LP1+b*v);
}

// Returns the shortest distance from a point P to a line segment
// defined by two points (SP1 and SP2)
__device__ float point_seg_dist(float3 P, float3 SP1, float3 SP2){
    float3 v = SP2-SP1;
    float c1 = dot(P-SP1, v);
    if(c1<=0) return d(P, SP1);
    float c2 = dot(v, v);
    if(c2<=c1) return d(P, SP2);
    float3 Pb = SP1 + c1/c2 * v;
    return d(P, Pb);
}

__device__ void boxMuller(float& u1, float& u2){ // num of ops = 5
    float   r = sqrtf(-2.0f * __logf(u1));
    float phi = TWOPI * u2;
    u1 = r * __cosf(phi);
    u2 = r * __sinf(phi);
}

/*__device__ uint rand31pmc(uint &seed){//num of ops = 5
   uint hi, lo;
   lo = 16807 * (seed & 0xFFFF);
   hi = 16807 * (seed >> 16);
   lo += (hi & 0x7FFF) << 16;
   lo += hi >> 15;                  
   if (lo > 0x7FFFFFFF) lo -= 0x7FFFFFFF;          
   return ( seed = lo );        
}*/

/*
 * A faster 48-bit PNRG from Arnold and van Meel (released under GPL). 
  Copyright (c) 2007 A. Arnold and J. A. van Meel, FOM institute
  AMOLF, Amsterdam; all rights reserved unless otherwise stated.
  "Harvesting graphics power for MD simulations"
  by J.A. van Meel, A. Arnold, D. Frenkel, S. F. Portegies Zwart and
  R. G. Belleman, arXiv:0709.3225.
 
 * propagate an rand48 RNG one iteration.
    @param Xn  the current RNG state, in 2x 24-bit form
    @param A,C the magic constants for the RNG. For striding,
               this constants have to be adapted, see the constructor
    @result    the new RNG state X(n+1)
*/
/*__device__
static uint2 RNG_rand48_iterate_single(uint2 Xn, uint2 A, uint2 C){
  // results and Xn are 2x 24bit to handle overflows optimally, i.e.
  // in one operation.

  // the multiplication commands however give the low and hi 32 bit,
  // which have to be converted as follows:
  // 48bit in bytes = ABCD EF (space marks 32bit boundary)
  // R0             = ABC
  // R1             =    D EF

  unsigned int R0, R1;

  // low 24-bit multiplication
  const unsigned int lo00 = __umul24(Xn.x, A.x);
  const unsigned int hi00 = __umulhi(Xn.x, A.x);

  // 24bit distribution of 32bit multiplication results
  R0 = (lo00 & 0xFFFFFF);
  R1 = (lo00 >> 24) | (hi00 << 8);

  R0 += C.x; R1 += C.y;

  // transfer overflows
  R1 += (R0 >> 24);
  R0 &= 0xFFFFFF;

  // cross-terms, low/hi 24-bit multiplication
  R1 += __umul24(Xn.y, A.x);
  R1 += __umul24(Xn.x, A.y);

  R1 &= 0xFFFFFF;

  return make_uint2(R0, R1);
}*/


__device__ uint myRand(uint seed[]){//num of ops = 5
   // Simple multiply-with-carry PRNG that uses two seeds (seed[0] and seed[1])
   // (Algorithm from George Marsaglia: http://en.wikipedia.org/wiki/George_Marsaglia)
    seed[0] = 36969 * (seed[0] & 65535) + (seed[0] >> 16);
    seed[1] = 18000 * (seed[1] & 65535) + (seed[1] >> 16);
    return (seed[0] << 16) + seed[1];
}

/* 
 * Return a random number r in the range 0<=r<=1
 */
__device__ float myRandf(uint seed[]){
    return((float)myRand(seed) / 4294967295.0f);
}

/* 
 * Return a vector with a specified magnitude (adc) and a random direction.
 */
__device__ void myRandDir(uint seed[], float adc, float3& vec){
    // azimuth and elevation are on the interval [0,2*pi)
    // (2*pi)/4294967294.0 = 1.4629181e-09f
    float az = (float)myRand(seed) * 1.4629181e-09f;
    float el = (float)myRand(seed) * 1.4629181e-09f;
    vec.z = adc * __sinf(el);
    float rcosel = adc * __cosf(el);
    vec.x = rcosel * __cosf(az);
    vec.y = rcosel * __sinf(az);
    return;
}

/*
 * returns three random numbers from the normal distribution (mean 0, std 1) and 
 * a forth from the uniform distribution.
 */
__device__ void myRandn(uint seed[], float& n1, float& n2, float& n3, float& u) {//num of ops = 8 + 4*5 + 5*2
  // We want random numbers in the range (0,1] (i.e., 0>n>=1):
  n1 = ((float)myRand(seed) + 1.0f) / 4294967296.0f;
  n2 = ((float)myRand(seed) + 1.0f) / 4294967296.0f;
  n3 = ((float)myRand(seed) + 1.0f) / 4294967296.0f;
  u  = ((float)myRand(seed) + 1.0f) / 4294967296.0f;
  // Note that ULONG_MAX=4294967295
  float n4 = u;
  boxMuller(n1, n2);
  boxMuller(n3, n4);
  return;
}
	
// calculate position in uniform cube
__device__ int3 calcCubePos(float3 p,
                            float cubeLength
                            )
{//num of ops = 6
    int3 cubePos;
    cubePos.x = floor((p.x + 1.0f) / cubeLength);
    cubePos.y = floor((p.y + 1.0f) / cubeLength);
    cubePos.z = floor((p.z + 1.0f) / cubeLength);
    return cubePos;
}

// calculate address in cube from position (clamping to edges)

__device__ uint calcCubeHash(int3 cubePos,
                             uint numCubes)
{//num of ops = 8
    cubePos.x = max(0, min(cubePos.x, numCubes-1));
    cubePos.y = max(0, min(cubePos.y, numCubes-1));
    cubePos.z = max(0, min(cubePos.z, numCubes-1));
    return cubePos.z * numCubes * numCubes + cubePos.y* numCubes + cubePos.x;
}

// calculate position in uniform cube
__device__ int3 calcCubePos_4(float4 p,
                            float cubeLength
                            )
{//num of ops = 6
    int3 cubePos;
    cubePos.x = floor((p.x + 1.0f) / cubeLength);
    cubePos.y = floor((p.y + 1.0f) / cubeLength);
    cubePos.z = floor((p.z + 1.0f) / cubeLength);
    return cubePos;
}


/*
* Function: integrate()
* Return type: void
* Description: Computes the spin movement.
*
* oldPos is an array of float4's, where oldPos[i].x,y,z = 3d spatial coords of 
* spin i and oldPos[i].w is the phase of spin i.
*
* oldSeed is an array of uint4's where oldSeed[i].x,y are the two PRNG seeds 
* for spin i. oldSeed[i].
*
*/
__global__ void
integrate(  float4* oldPos, 
            uint4* oldSeed,
            float deltaTime,
            float permeability,
            float intraStdDev,
            float extraStdDev,
	    float myelinStdDev,          
            int numBodies,
            float gradX, float gradY, float gradZ,
            float4* fiberPos_1,
            uint* cubeCounters,
            uint* cubeList,
            float phaseConstant,
            float cubeLength,
            uint numCubes,
            uint maxFibersPerCube,
            float innerRadiusScale,
            uint iterations){
            
   int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

   if(index>=numBodies)
   return;

   float phase = oldPos[index].w;    
   float3 pos = make_float3(oldPos[index]);    
   //oldPos[index] = make_float4(pos.x+0.001f,pos.y-0.001f,pos.z+0.001f,phase);

   uint4 seed4 = oldSeed[index];
   uint insideFiberIndex = seed4.w;

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

   uint rseed[2];
   rseed[0] = seed4.x;
   rseed[1] = seed4.y; // + clock() + (uint)index + 1000;
   
   bool isInside, reflected;
   
   for(uint i=0; i<iterations; i++){
      isInside = insideFiberIndex<UINT_MAX;
      // Take a random walk...
      // myRandn returns 3 PRNs from a normal distribution with mean 0 and SD of 1. 
      // So, we just need to scale these with the desired SD to get the displacements
      // for the random walk.
      // myRandn also returns a bonus uniformly distributed PRN as a side-effect of the 
      // Box-Muller transform used to generate normally distributed PRNs.
      float u;
      float3 vel;
      ///*
      float3 brnMot;
      myRandn(rseed, brnMot.y, brnMot.x, brnMot.z, u);
      if(isInside)
         vel = brnMot * intraStdDev;
      else
         vel = brnMot * extraStdDev;
      ///*
      /*
      if(isInside)
         myRandDir(rseed, intraStdDev, vel);
      else
         myRandDir(rseed, extraStdDev, vel);
      u = myRandf(rseed);
      */

      pos += vel;
      //--->num of ops untill now in the kernel = 40

      // don't let the spin leave the volume
      if (pos.x > 1.0f)  { pos.x = 1.0f; }
      else if (pos.x < -1.0f) { pos.x = -1.0f; }
      if (pos.y > 1.0f)  { pos.y = 1.0f; }
      else if (pos.y < -1.0f) { pos.y = -1.0f; }
      if (pos.z > 1.0f)  { pos.z = 1.0f; }
      else if (pos.z < -1.0f) { pos.z = -1.0f; }
   
      if(permeability<1.0f){
         reflected = false;
         float curDistToCenter;
         float3 posVec;
         float3 bounceVec = {-1.0f, -1.0f, -1.0f};

         // TO DO:
         // take account of innerRadiusScale (myelin sheath). 
         // Maybe compute the vector representing the spin position relative to the 
         // 
         if(isInside){
            float4 fiberTmp = tex1Dfetch(texFiberList, insideFiberIndex);
            float3 fiberPos = make_float3(fiberTmp);
            float fiberRadSq = fiberTmp.w;
            
            // fiberPos.x|y|z == 2 means this is the longitudial axis of the fiber
            if(fiberPos.x==2.0f)     { fiberPos.x = pos.x; bounceVec.x = 0.0f; }
            else if(fiberPos.y==2.0f){ fiberPos.y = pos.y; bounceVec.y = 0.0f; }
            else if(fiberPos.z==2.0f){ fiberPos.z = pos.z; bounceVec.z = 0.0f; }
            posVec = fiberPos-pos;
            curDistToCenter = dot(posVec,posVec);
            if(curDistToCenter>=fiberRadSq*innerRadiusScale){
               if(u>=permeability) reflected = true;
               else insideFiberIndex = UINT_MAX;
            }//num of ops = 5 for this
         }else{
            uint cubeIndex;

            cubeIndex = calcCubeHash(calcCubePos(pos,cubeLength), numCubes);   		      

            for(uint j=0;j<cubeCounters[cubeIndex];j++){
					uint curFiberIndex = tex1Dfetch(texCubeList,cubeIndex*maxFibersPerCube+j);
               float4 fiberTmp = tex1Dfetch(texFiberList, curFiberIndex);
               float3 fiberPos = make_float3(fiberTmp);
               float fiberRadSq = fiberTmp.w;
               
               if(fiberPos.x==2.0f)     { fiberPos.x = pos.x; bounceVec.x = 0.0f; }
               else if(fiberPos.y==2.0f){ fiberPos.y = pos.y; bounceVec.y = 0.0f; }
               else if(fiberPos.z==2.0f){ fiberPos.z = pos.z; bounceVec.z = 0.0f; }
               posVec = fiberPos-pos;
               curDistToCenter = dot(posVec,posVec);

               if(curDistToCenter<=fiberRadSq){                  
                  if(u>=permeability) reflected = true;
                  else insideFiberIndex = curFiberIndex;
                  break;//can break once a fiber interaction is detected
               }
            }//num of ops = 8 * 25 [38 max] (avg case)
         }

         if(reflected){
            // TO DO: approximate a bounce here!
            pos = pos+vel*bounceVec; 
         }  
      } // end if(permeability)
      
      // calculate the local magnetic field of each spin and adjust the spin phase accordingly
      phase += (gradX * pos.x + gradY * pos.y + gradZ * pos.z) * phaseConstant; //num of ops = 7
   }
   
   // store new position and velocity
   oldPos[index] = make_float4(pos,phase);
   // store new seed values
   oldSeed[index].x = rseed[0];
   oldSeed[index].y = rseed[1];
   oldSeed[index].w = insideFiberIndex; // num of ops = 5
   //TOTAL OPS for the kernel = 40 + 200 + 6 = 246 * niter 
}



__global__ void
integrateTest(  float4* oldPos, 
            float adcStdDev,          
            int numBodies,
            float gradX, float gradY, float gradZ,
            uint iterations){
            
   int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

   if(index>=numBodies)
   return;

   float phase = oldPos[index].w;    
   float3 pos = make_float3(oldPos[index]);    

   float3 brnMot;
   uint rseed[2];
   rseed[0] = clock() + (uint)index;
   rseed[1] = rseed[0] + 1234567;
   
   for(uint i=0; i<iterations; i++){
      float u;
      myRandn(rseed, brnMot.y, brnMot.x, brnMot.z, u);
      pos += brnMot * adcStdDev;

      // bounce off cube sides
      if (pos.x > 1.0f)  { pos.x = 1.0f; }
      if (pos.x < -1.0f) { pos.x = -1.0f; }
      if (pos.y > 1.0f)  { pos.y = 1.0f; }
      if (pos.y < -1.0f) { pos.y = -1.0f; }
      if (pos.z > 1.0f)  { pos.z = 1.0f; }
      if (pos.z < -1.0f) { pos.z = -1.0f; }
   }
   
   // store new position and velocity
   oldPos[index] = make_float4(pos,phase);
}


#endif
