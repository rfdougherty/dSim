/*
 *  This file contains the implementation of the functions defined in spinSystem.h
 */

#include "spinSystem.h"
#include "spinSystem.cuh"
#include "spinKernel.cuh"
#include "radixsort.cuh"
#include "options.h"
#include <cutil.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>
#include <sys/time.h>
#include <limits.h>

#include <iostream>

#ifndef CUDART_PI_F
#define CUDART_PI_F 3.141592654f
#endif
#define PI 3.14159265358979f
#define TWOPI 6.28318530717959f

using namespace std;

//just the usual max, min
template<class x>
x max(x a,x b)
{
   if(a<b) return b;
    return a;
}

template<class x>
x min(x a,x b)
{
   if(a>b) return b;
    return a;
}

inline float frand(){
    return rand() / (float) RAND_MAX;
}


//
// Computes the intersection of a line segment (defined by SP1 and SP2) 
// with a plane (defined by PP, a point in the plane, and PN, a point on 
// the normal to the plane). 
//
//   Output: IP = the intersect point (when it exists)
//   Returns: 0 = for no intersection
//            1 = unique intersection at the point IP
//            2 = segment lies in the plane
//
inline int intersectLinePlane(float3 SP1, float3 SP2, float3 PP, float3 PN, float3 &IP)
{
    float3 u = SP2 - SP1;
    float3 w = SP2 - PP;

    float D = dot(PN, u);
    float N = -dot(PN, w);

    if(fabs(D) < 0.0000001) {     
        // segment is parallel to plane, so it can either:
        if(N==0)          // lie in plane
            return 2;
        else              // no intersection
            return 0;
    }
    // not parallel- check for intersection
    float sI = N / D;
    if (sI < 0 || sI > 1)
        return 0;           // no intersection

    IP = SP1 + sI * u;     // segment intersect point
    return 1;
}


/*
 * SpinSystem Constructor- just initializes the key imutable params.
 * Call the build method to allocate memory and initFibers to create the cellular
 * entites.
 */
SpinSystem::SpinSystem(int numSpins, bool useGpu, float spaceScale, float gyromagneticRatio, bool useVbo):
        m_bInitialized(false)
{
    m_nParams = 5;
    m_numSpins = numSpins;
    m_hPos = 0;
    m_hPartParams = 0;
    m_hSeed = 0;
    m_currentPosRead = 0;
    m_currentSeedRead = 0;
    m_currentPosWrite = 1;
    m_currentSeedWrite = 1;
    // The following is a hard limit. If the actual 'entities per cube' exceeds 
    // m_maxFibersPerCube, you will get a segFault or (worse), inaccurate results.
    m_maxFibersPerCube = 600;
    m_useGpu = useGpu;
    m_spaceScale = spaceScale;
    m_gyromagneticRatio = gyromagneticRatio;
    m_useVbo = useVbo;
    m_timer = 0;

    m_dPos[0] = m_dPos[1] = 0;
    m_dSeed[0] = m_dSeed[1] = 0;
    m_spinRadius = 0.005; // note that this is only used for rendering
    m_totalNumTimeSteps = 0;

    //m_T2fiber = 80;
    //m_T2myelin = 7.5;
    //m_T2outside = 80;
    
    // Seed the PRNG. This is used to initialize spin postions and to set
    // the inital PRNG seed values for the GPU kernel.
    //srand(1973);
    srand(time(0));

    //printf("m_spinRadius: %g \n", m_spinRadius);

}

/*
 *  Allocates memory for arrays in host and device
 * 
 */
bool SpinSystem::build()
{
   // Only allow initialization once.
    assert(!m_bInitialized);
    checkCUDA();

    if(m_maxFiberRadius<=0.0) return(false);

    uint tempGridSize = (uint)(2.0f/(float)(m_maxFiberRadius*4))-1;
    if(tempGridSize<2) tempGridSize = 2;
    if(tempGridSize>MAX_GRID_SIZE){
        printf("WARNING: grid size clipped to MAX_GRID_SIZE (%d)- results may be inaccurate!\n",MAX_GRID_SIZE);
        tempGridSize = MAX_GRID_SIZE;
    }
    ///number of cubes in any direction
    m_numCubes = tempGridSize;
    //total number of cubes in the world
    m_totalNumCubes = m_numCubes*m_numCubes*m_numCubes;
	printf("m_numCubes: %u\n", m_numCubes);

    m_maxFibersPerCube = ceil(m_numFibers/(m_numCubes*m_numCubes)*10);

    //size of each cube/cube in the world
    m_cubeLength = 2.0f / m_numCubes;

    // allocate host storage
    m_hPos = new float[m_numSpins*4];					// Change 1/27: Changed from 4 to 5
    m_hPartParams = new float[m_numSpins*m_nParams];			// Change 2/4: Added line
    m_hSeed = new uint[m_numSpins*4];					// Change 1/27: Changed from 4 to 5
    memset(m_hPos, 0, m_numSpins*4*sizeof(float));			// Change 1/27: Changed from 4 to 5
    memset(m_hPartParams, 0, m_numSpins*m_nParams*sizeof(float));	// Change 2/4: Added line
    memset(m_hSeed, 0, m_numSpins*4*sizeof(uint));			// Change 1/27: Changed from 4 to 5

    m_hCubeCounter = new uint[m_totalNumCubes];
    m_hCubes = new uint[m_totalNumCubes*m_maxFibersPerCube];

    memset(m_hCubeCounter, 0, m_totalNumCubes*sizeof(uint));
    
    int nGridBytes = m_totalNumCubes*m_maxFibersPerCube*sizeof(uint);
    printf("cube size: %d with %d total cubes, requiring %0.2g MB of host storage and texture memory.\n", m_numCubes, m_totalNumCubes, (float)nGridBytes/1048576);
    //this holds the index of the fiber in the fiber array:
    memset(m_hCubes, 0, nGridBytes);

    // allocate GPU data
    if(m_useVbo){
printf("test\n");
        m_posVbo[0] = createVBO(sizeof(float) * 4 * m_numSpins);		// Change 1/27: Changed from 4 to 5
        m_posVbo[1] = createVBO(sizeof(float) * 4 * m_numSpins);		// Change 1/27: Changed from 4 to 5
    }else{
        allocateArray((void**)&m_dPos[0], sizeof(float) * 4 * m_numSpins);	// Change 1/27: Changed from 4 to 5
        allocateArray((void**)&m_dPos[1], sizeof(float) * 4 * m_numSpins);	// Change 1/27: Changed from 4 to 5
    }

    allocateArray((void**)&m_dSeed[0], sizeof(uint) * 4 * m_numSpins);		// Change 1/27: Changed from 4 to 5
    allocateArray((void**)&m_dSeed[1], sizeof(uint) * 4 * m_numSpins);		// Change 1/27: Changed from 4 to 5

	//freeArray(m_dSeed[0]);				// Testing freeArray		
	//freeArray(m_dSeed[1]);				// Testing freeArray
	//std::cin.get();

    allocateArray((void**)&m_dFiberPos, sizeof(float) * 4 * m_numFibers);

    // populate the system with fibers
    if(!addEntities()) return(false);

    copyArrayToDevice(m_dFiberPos, m_fiberPos, 0, sizeof(float) * 4 * m_numFibers);
    bindFiberList(m_dFiberPos, m_numFibers);

    allocateArray((void**)&m_dCubeCounter, m_totalNumCubes*sizeof(uint));

    copyArrayToDevice(m_dCubeCounter, m_hCubeCounter, 0, m_totalNumCubes*sizeof(uint));
    allocateArray((void**)&m_dCubes, nGridBytes);
    copyArrayToDevice(m_dCubes, m_hCubes, 0, nGridBytes);
	bindCubeList(m_dCubes, m_totalNumCubes*m_maxFibersPerCube);

    if(m_useVbo){
        m_colorVBO = createVBO(m_numSpins*4*sizeof(float));
        // fill color buffer
        glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
        float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        float *ptr = data;
        for(uint i=0; i<m_numSpins; i++) {
            *ptr++ = 0.5f;
            *ptr++ = 0.5f;
            *ptr++ = 0.5f;
	    *ptr++ = 0.5f;//for spin
        }
        glUnmapBufferARB(GL_ARRAY_BUFFER);
    }

    CUT_SAFE_CALL(cutCreateTimer(&m_timer));
    m_bInitialized = true;
    printf("Finished initializing spin system\n");
    return(true);
}


/*
 * Set up the positions/radius of the cellular entities from parameters.
 */
bool SpinSystem :: initFibers(float fiberRadius, float fiberRadiusStd, float fiberSpace, float fiberSpaceStd, float fiberInnerRadiusProportion, float fiberCrossProportion)
{
    m_innerRadiusProportion = fiberInnerRadiusProportion;
    float r, maxRadius;
    fiberRadius = fiberRadius/m_spaceScale;
    fiberSpace = fiberSpace/m_spaceScale;
    fiberRadiusStd = fiberRadiusStd/m_spaceScale;
    fiberSpaceStd = fiberSpaceStd/m_spaceScale;    
    float xSpace = 2*fiberRadius+fiberSpace;
    
    // We use hexagonal packing. To do this, we simply stagger each row by the
    // half the center-spacing 'xSpace' (= 2*fiberRadius + fiberSpace). However, each 
    // row is spaced a little less than xSpace. In fact, the centers of any 3 fibers 
    // form an equilateral triangle, so the row distance is the altitude of the 
    // equilateral triangle with a side length of xSpace. This is sqrt(3)/2*xSpace.
    float ySpace = sqrt(3.0f)/2.0f*xSpace;
    float halfXSpace = 0.5f*xSpace;
    float crossSpaceOffset = 0.0f;
    int nRows = (int)(2.0/ySpace+0.5);
    int nCols = (int)(2.0/xSpace+0.5);
   
    m_numFibers = nRows*nCols;
    printf("Generating %d fibers...\n", m_numFibers);
   
    m_fiberPos = new float[m_numFibers*4];
    maxRadius = 0.0f;
    int crossRow = (int)(fiberCrossProportion*(float)nRows+0.5);
    for(int i=0; i<nRows; i++){
        for(int j=0; j<nCols; j++){
            int index = i*nCols*4+j*4;
            if(i==crossRow) crossSpaceOffset = xSpace-ySpace;
            if(i<crossRow){
               // Set the x-coordinate
               if(i&1) m_fiberPos[index] = -1.0f + (float)j*xSpace + halfXSpace;
               else    m_fiberPos[index] = -1.0f + (float)j*xSpace;
               // Set the y coordinate
               m_fiberPos[index+1] = -1.0f + (float)i*ySpace;
               if(fiberSpaceStd>0){
                   m_fiberPos[index  ] += fiberSpaceStd*(frand()*2.0f-1.0f);
                   m_fiberPos[index+1] += fiberSpaceStd*(frand()*2.0f-1.0f);
               }

               // a value of 2 signals that this is a fiber and marks the longitudinal axis
               m_fiberPos[index+2] = 2.0f;
            }else{
            // *** WORK HERE: confirm this and ensure there is no overlap.
               // Set the y-coordinate
               if(i&1) m_fiberPos[index+1] = -1.0f + (float)j*xSpace + halfXSpace;
               else    m_fiberPos[index+1] = -1.0f + (float)j*xSpace;
               // Set the z coordinate
               m_fiberPos[index+2] = -1.0f + (float)i*ySpace + crossSpaceOffset;
               if(fiberSpaceStd>0){
                   m_fiberPos[index+1] += fiberSpaceStd*(frand()*2.0f-1.0f);
                   m_fiberPos[index+2] += fiberSpaceStd*(frand()*2.0f-1.0f);
               }

               // a value of 2 signals that this is a fiber and marks the longitudinal axis
               m_fiberPos[index] = 2.0f;
            }

      /*
            if(i%2==0 &&j%2!=0){
            float temp = m_fiberPos[(i+n)*n2*4+(j+n)*4+2];
            m_fiberPos[(i+n)*n2*4+(j+n)*4+2] = m_fiberPos[(i+n)*n2*4+(j+n)*4  ];
            m_fiberPos[(i+n)*n2*4+(j+n)*4  ] = temp;
        }
       
            if(i%2==0 &&j%2==0){
            float temp = m_fiberPos[(i+n)*n2*4+(j+n)*4+2];
            m_fiberPos[(i+n)*n2*4+(j+n)*4+2] = m_fiberPos[(i+n)*n2*4+(j+n)*4 +1 ];
            m_fiberPos[(i+n)*n2*4+(j+n)*4 +1  ] = temp;
        }
      */
         // Set the radius
            r = fiberRadius;
            if(fiberRadiusStd>0) r += fiberRadiusStd*(frand()*2.0f-1.0f);
         // To save a multiply in the kernel, we stored the squared radius
            m_fiberPos[index+3] = r*r;

            if(r>maxRadius) maxRadius = r;
        }
    }
    m_maxFiberRadius = maxRadius;
    return(true);
}

/*
 * Load the positions/radius of the cellular entities from a file.
 */
bool SpinSystem :: initFibers(FILE *fiberFilePtr, float fiberInnerRadiusProportion)
{
    m_innerRadiusProportion = fiberInnerRadiusProportion;
    float x,y,z,r;

    if(!fiberFilePtr) return(false);
   
    // Count the fibers
    printf("Counting fibers...\n");
	printf("fiberInnerRadiusProportion: %g\n", fiberInnerRadiusProportion);
    m_numFibers = 0;
    //char *str = (char *)malloc(256*sizeof(char));
    char str[256];
    while(fgets(str,255,fiberFilePtr)!=NULL){
        //printf(str);
        if(sscanf(str, "%f %f %f %f\n",&x,&y,&z,&r)==4 
           && (fabs(x)+r<m_spaceScale || x==INFINITY)
           && (fabs(y)+r<m_spaceScale || y==INFINITY)
           && (fabs(z)+r<m_spaceScale || z==INFINITY)
           && r>0)
            m_numFibers++;
    }
    printf("Reading %d fibers...\n",m_numFibers);
    // Now read the file again, extracting the values
    rewind(fiberFilePtr);
    m_fiberPos = new float[m_numFibers*4];
    float *fpInd = m_fiberPos;
    m_maxFiberRadius = 0.0f;
    printf("Loading %d fibers...\n", m_numFibers);
//	float totalArea = 0;										// Temp line
    while(fgets(str,256,fiberFilePtr)!=NULL){
        if(sscanf(str, "%f %f %f %f",&x,&y,&z,&r)==4
           && (fabs(x)+r<m_spaceScale || x==INFINITY)
           && (fabs(y)+r<m_spaceScale || y==INFINITY)
           && (fabs(z)+r<m_spaceScale || z==INFINITY)
           && r>0){
            //printf("unscaled = [%f %f %f %f]; ",x,y,z,r);
            r = r/m_spaceScale;
            if(r>m_maxFiberRadius) m_maxFiberRadius = r;
            x = x/m_spaceScale;
            if(x>2.0||x==INFINITY) x = 2.0;
            y = y/m_spaceScale;
            if(y>2.0||y==INFINITY) y = 2.0;
            z = z/m_spaceScale;
            if(z>2.0||z==INFINITY) z = 2.0;
            //printf("scaled = [%f %f %f %f]\n",x,y,z,r);
            *(fpInd++) = x;
            *(fpInd++) = y;
            *(fpInd++) = z;
            // To save a multiply in the kernel, we stored the squared radius
            *(fpInd++) = r*r;
//		totalArea = totalArea + r*r;								// Temp line
        }
    }
//	printf("Total area: %g \n", totalArea);								// Temp line
    printf("maxFiberRadius = %0.2f (%0.4f scaled)\n", m_maxFiberRadius*m_spaceScale, m_maxFiberRadius);
	printf("in initFibers: m_fiberPos[0] = %g\n", m_fiberPos[0]);
	printf("in initFibers: m_fiberPos[1] = %g\n", m_fiberPos[1]);
	printf("in initFibers: m_fiberPos[2] = %g\n", m_fiberPos[2]);
	printf("in initFibers: m_fiberPos[3] = %g\n", m_fiberPos[3]);
//	cin.get();											// Temp line
    return(true);
}


SpinSystem::~SpinSystem()
{
    _finalize();
    m_numSpins = 0;
}

GLuint
SpinSystem::createVBO(uint size)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    registerGLBufferObject(vbo);
    return vbo;
}

/*
 * Function: calcCubePos()
 * Return type: float3
 * Parameters: float3 point,
 * Description: Function calculates the cube cell to which the given position belongs in uniform cube
 */

float3 SpinSystem::calcCubePos(float3 point)
{
    float3 cubePos = floor((point + 1.0f) / m_cubeLength);
    return cubePos;
}


/*
 * Function: calcCubeHash()
 * Return type: int
 * Parameters: float3 cubePos
 * Description: Function calculates address in cube (index in list of cells) from position (clamping to edges)
 */
int SpinSystem::calcCubeHash(float3 cubePos)
{
   cubePos = fmaxf(0.0f, fminf(m_numCubes-1, cubePos));		// m_numCubes is the number of cubes in any one dimension

   // printf("(%f,%f,%f) ",cubePos.x,cubePos.y,cubePos.z);
   return  ( (int)(m_numCubes*m_numCubes*cubePos.z) + (int)(m_numCubes*cubePos.y) + (int)cubePos.x);   	// Cant take m_numCubes^3 values, from 0 to m_numCubes^3-1

}


/*
 *  Function: addEntities()
 *  Return Type: void
 *  Parameters: NONE
 *  Description: Function populates the list that contains the entities present in each cube Cell in the voxel
 *  initializing the fibers are done by:
 *  1. run through the fiber list and grab the co-ordinates for each fiber,
 *  2. given the initial (x,y,z) add the fiber entity to all the numCubes.z cells by 
 *     - hash into the cell list to get appropriate cell to which the fiber position is to be included
 *     - add 'cubeLength' to the z co-ordinate to get yet another position to hash
 */
//Macro defined to keep code size low
#define FILL_LIST_Z {\
         currPoint.z = -1 + 0.002;\
         for(uint j=0;j<m_numCubes;j++){\
            float3 cubePos;\
            cubePos = calcCubePos(currPoint);\
            int index = calcCubeHash(cubePos);\
            int counter = m_hCubeCounter[index];\
            m_hCubeCounter[index]+=1;\
            m_hCubes[index*m_maxFibersPerCube+counter] = i/4;\
          	currPoint.z += m_cubeLength;\
         }\
         }
#define FILL_LIST_X {\
			currPoint.x = -1 + 0.002;\
         for(uint j=0;j<m_numCubes;j++){\
            int index = calcCubeHash(calcCubePos(currPoint));\
            int counter = m_hCubeCounter[index];\
            m_hCubeCounter[index]+=1;\
            m_hCubes[index*m_maxFibersPerCube+counter] = i/4;\
            currPoint.x += m_cubeLength;\
         }\
         }
#define FILL_LIST_Y {\
         currPoint.y = -1 + 0.002;\
         for(uint j=0;j<m_numCubes;j++){\
            float3 cubePos;\
            cubePos = calcCubePos(currPoint);\
            int index = calcCubeHash(cubePos);\
            int counter = m_hCubeCounter[index];\
            m_hCubeCounter[index]+=1;\
            m_hCubes[index*m_maxFibersPerCube+counter] = i/4;\
            currPoint.y += m_cubeLength;\
        }\
        }
        
bool SpinSystem::addEntities(){
   uint maxEntities=0;
   for(uint i=0;i<m_totalNumCubes;i++)
    m_hCubeCounter[i] = 0;
   #ifdef FILE_WRITE
   FILE *fptr;
   fptr = fopen("cellList.txt","w");
   #endif

 
 	for(uint i=0;i<m_numFibers*4;i+=4){
      float3 currPoint; 
      currPoint.x = m_fiberPos[i];
      currPoint.y = m_fiberPos[i+1];
      currPoint.z = m_fiberPos[i+2];
      float3 tempTubePos;

      //printf("(%f, %f, %f)",currPoint.x,currPoint.y,currPoint.z); 
      if(m_fiberPos[i] == 2.0f){ //its a fiber oriented on the x-axis
         //skewing a bit to land fully inside the cell, for the fiber the z-coordinate is insignificant as it an infinitely long fiber 
         currPoint.x = -1 + 0.002; 
         for(uint j=0;j<m_numCubes;j++){
            int index = calcCubeHash(calcCubePos(currPoint));
            int counter = m_hCubeCounter[index];

            m_hCubeCounter[index]+=1;
            #ifdef FILE_WRITE 
            fprintf(fptr,"\n(fiber - %d, x - %d, cell - %d, counter - %d),",i,j,index, counter);    
            #endif
            m_hCubes[index*m_maxFibersPerCube+counter] = i/4;//storing the index of the fiber, indicating that this entity is in this cube
            currPoint.x += m_cubeLength; //move to the next cube in the same line (x-axis)
         }
			//to check for an neighboring cubes to which the fiber might belong
			//start from down - curr - up
			tempTubePos.x = -1 + 0.002;
			tempTubePos.y = m_fiberPos[i+1];
			tempTubePos.z = m_fiberPos[i+2];
			
			float3 c_cubePos;
			c_cubePos = calcCubePos(tempTubePos);
   		float3 distanceFromCorner;

   		distanceFromCorner.y = tempTubePos.y - c_cubePos.y * m_cubeLength;   					
   		distanceFromCorner.z = tempTubePos.z - c_cubePos.z * m_cubeLength;
   		   		
			if(distanceFromCorner.z < HALO_DISTANCE * m_maxFiberRadius){
				currPoint.z -= HALO_DISTANCE * m_maxFiberRadius;
				if(currPoint.z > -1){
					FILL_LIST_X				
					if( (currPoint.y - m_cubeLength) > -1){
						currPoint.y -= m_cubeLength;
						FILL_LIST_X
						currPoint.y += m_cubeLength;
					}
					if( (currPoint.y + m_cubeLength) < 1){
						currPoint.y += m_cubeLength;
						FILL_LIST_X
						currPoint.y -= m_cubeLength;
					}
				}
				currPoint.z += HALO_DISTANCE * m_maxFiberRadius;
			} 
			
			else if(distanceFromCorner.z > (m_cubeLength - HALO_DISTANCE * m_maxFiberRadius)){
				currPoint.z += HALO_DISTANCE * m_maxFiberRadius;
				if(currPoint.z < 1){
					FILL_LIST_X				
					if( (currPoint.y - m_cubeLength) > -1){
						currPoint.y -= m_cubeLength;
						FILL_LIST_X
						currPoint.y += m_cubeLength;
					}
					if( (currPoint.y + m_cubeLength) < 1){
						currPoint.y += m_cubeLength;
						FILL_LIST_X
						currPoint.y -= m_cubeLength;
					}
				}				
				currPoint.z -= HALO_DISTANCE * m_maxFiberRadius;
			}   		
			if(distanceFromCorner.y < HALO_DISTANCE * m_maxFiberRadius){
				if( (currPoint.y - m_cubeLength) > -1){
					currPoint.y -= m_cubeLength;
					FILL_LIST_X
					currPoint.y += m_cubeLength;
				}			
			} 
			else if(distanceFromCorner.y > (m_cubeLength - HALO_DISTANCE * m_maxFiberRadius)){
				if( (currPoint.y + m_cubeLength) < 1){
					currPoint.y += m_cubeLength;
					FILL_LIST_X
					currPoint.y -= m_cubeLength;
				}			
			} 
      }else if(m_fiberPos[i+2] == 2.0f){ //its a fiber oriented on the z-axis
         currPoint.z = -1 + 0.002;
         for(uint j=0;j<m_numCubes;j++){
            float3 cubePos;
            cubePos = calcCubePos(currPoint);
            int index = calcCubeHash(cubePos);
            int counter = m_hCubeCounter[index];

            m_hCubeCounter[index]+=1;
            #ifdef FILE_WRITE 
            fprintf(fptr,"\n(fiber - %d, z - %d, cell - %d, counter - %d),",i,j,index, counter);    
            #endif
            m_hCubes[index*m_maxFibersPerCube+counter] = i/4;
          	currPoint.z += m_cubeLength;
         }
			//to check for an neighboring cubes to which the fiber might belong
			//start from down - curr - up
			tempTubePos.x = m_fiberPos[i];
			tempTubePos.y = m_fiberPos[i+1];
			tempTubePos.z = -1 + 0.002;
			
			float3 c_cubePos;
			c_cubePos = calcCubePos(tempTubePos);
   		float3 distanceFromCorner;
   		
   		distanceFromCorner.x = tempTubePos.x - c_cubePos.x * m_cubeLength;
   		distanceFromCorner.y = tempTubePos.y - c_cubePos.y * m_cubeLength;   					
   		
			if(distanceFromCorner.x < HALO_DISTANCE * m_maxFiberRadius){
				currPoint.x -= HALO_DISTANCE * m_maxFiberRadius;
				if(currPoint.x > -1){
					FILL_LIST_Z				
					if( (currPoint.y - m_cubeLength) > -1){
						currPoint.y -= m_cubeLength;
						FILL_LIST_Z
						currPoint.y += m_cubeLength;
					}
					if( (currPoint.y + m_cubeLength) < 1){
						currPoint.y += m_cubeLength;
						FILL_LIST_Z
						currPoint.y -= m_cubeLength;
					}
				}
				currPoint.x += HALO_DISTANCE * m_maxFiberRadius;
			} 
			
			else if(distanceFromCorner.x > (m_cubeLength - HALO_DISTANCE * m_maxFiberRadius)){
				currPoint.x += HALO_DISTANCE * m_maxFiberRadius;
				if(currPoint.x < 1){
					FILL_LIST_Z				
					if( (currPoint.y - m_cubeLength) > -1){
						currPoint.y -= m_cubeLength;
						FILL_LIST_Z
						currPoint.y += m_cubeLength;
					}
					if( (currPoint.y + m_cubeLength) < 1){
						currPoint.y += m_cubeLength;
						FILL_LIST_Z
						currPoint.y -= m_cubeLength;
					}
				}				
				currPoint.x -= HALO_DISTANCE * m_maxFiberRadius;
			}   		
			if(distanceFromCorner.y < HALO_DISTANCE * m_maxFiberRadius){
				if( (currPoint.y - m_cubeLength) > -1){
					currPoint.y -= m_cubeLength;
					FILL_LIST_Z
					currPoint.y += m_cubeLength;
				}			
			} 
			else if(distanceFromCorner.y > (m_cubeLength - HALO_DISTANCE * m_maxFiberRadius)){
				if( (currPoint.y + m_cubeLength) < 1){
					currPoint.y += m_cubeLength;
					FILL_LIST_Z
					currPoint.y -= m_cubeLength;
				}			
			} 
      }else if(m_fiberPos[i+1] == 2.0f){ //its a fiber oriented on the y-axis
         currPoint.y = -1 + 0.002;
         for(uint j=0;j<m_numCubes;j++){
            float3 cubePos;
            cubePos = calcCubePos(currPoint);
            int index = calcCubeHash(cubePos);
            int counter = m_hCubeCounter[index];

            m_hCubeCounter[index]+=1;
            #ifdef FILE_WRITE 
            fprintf(fptr,"\n(fiber - %d, y - %d, cell - %d, counter - %d),",i,j,index, counter);    
            #endif
            m_hCubes[index*m_maxFibersPerCube+counter] = i/4;
            currPoint.y += m_cubeLength;
        }
  			//to check for an neighboring cubes to which the fiber might belong
			//start from down - curr - up
			tempTubePos.x = m_fiberPos[i];
			tempTubePos.y = -1 + 0.002;
			tempTubePos.z = m_fiberPos[i+2];
			
			float3 c_cubePos;
			c_cubePos = calcCubePos(tempTubePos);
   		float3 distanceFromCorner;
   		

   		distanceFromCorner.z = tempTubePos.z - c_cubePos.z * m_cubeLength;   					
   		distanceFromCorner.x = tempTubePos.x - c_cubePos.x * m_cubeLength;
   		   		
			if(distanceFromCorner.z < HALO_DISTANCE * m_maxFiberRadius){
				currPoint.z -= HALO_DISTANCE * m_maxFiberRadius;
				if(currPoint.z > -1){
					FILL_LIST_Y				
					if( (currPoint.x - m_cubeLength) > -1){
						currPoint.x -= m_cubeLength;
						FILL_LIST_Y
						currPoint.x += m_cubeLength;
					}
					if( (currPoint.x + m_cubeLength) < 1){
						currPoint.x += m_cubeLength;
						FILL_LIST_Y
						currPoint.x -= m_cubeLength;
					}
				}
				currPoint.z += HALO_DISTANCE * m_maxFiberRadius;
			} 
			
			else if(distanceFromCorner.z > (m_cubeLength - HALO_DISTANCE * m_maxFiberRadius)){
				currPoint.z += HALO_DISTANCE * m_maxFiberRadius;
				if(currPoint.z < 1){
					FILL_LIST_Y				
					if( (currPoint.x - m_cubeLength) > -1){
						currPoint.x -= m_cubeLength;
						FILL_LIST_Y
						currPoint.x += m_cubeLength;
					}
					if( (currPoint.x + m_cubeLength) < 1){
						currPoint.x += m_cubeLength;
						FILL_LIST_Y
						currPoint.x -= m_cubeLength;
					}
				}				
				currPoint.z-= HALO_DISTANCE * m_maxFiberRadius;
			}   		
			if(distanceFromCorner.x < HALO_DISTANCE * m_maxFiberRadius){
				if( (currPoint.x - m_cubeLength) > -1){
					currPoint.x -= m_cubeLength;
					FILL_LIST_Y
					currPoint.x += m_cubeLength;
				}			
			} 
			else if(distanceFromCorner.x > (m_cubeLength - HALO_DISTANCE * m_maxFiberRadius)){
				if( (currPoint.x + m_cubeLength) < 1){
					currPoint.x += m_cubeLength;
					FILL_LIST_Y
					currPoint.x -= m_cubeLength;
				}			
			}
      }
   }
   
   for(uint i=0;i<m_totalNumCubes;i++)
	   if(m_hCubeCounter[i]>maxEntities)
		   maxEntities = m_hCubeCounter[i]; 
   printf("\nActual maximum number of fibers per cell = %d (limit = %d)\n", maxEntities,m_maxFibersPerCube);
   #ifdef FILE_WRITE
   fclose(fptr);
   #endif
  // printOutCells();  
   if(maxEntities>m_maxFibersPerCube) return(false); 
   return(true); 
}


void SpinSystem:: printOutCells(){
   for(uint i=0;i<m_totalNumCubes;i++){
      if(m_hCubeCounter[i]!=0){
         printf("\nFor Cell %d: no. of entities - %d, fibers:",i,m_hCubeCounter[i]);
         for(uint j=0;j<m_hCubeCounter[i];j++){
	         printf(" %d", m_hCubes[i*4+j]);
         }
      }
   }      
}


/*
 *  Function: _finalize()
 *  Return Type: void
 *  Parameters: NONE
 *  Description: This function deallocates the memory used for arrays in host and device, basically the entire delete before program exit
 * 
 */
void SpinSystem::_finalize()
{
    assert(m_bInitialized);
    
    CUT_SAFE_CALL(cutDeleteTimer(m_timer));
    unbindFiberList();
    unbindCubeList();
    delete [] m_hPos;
    delete [] m_hSeed;
    delete [] m_fiberPos;
    delete [] m_hSpinHash;
    delete [] m_hCubeCounter;
    delete [] m_hCubes;

    freeArray(&m_dSeed[0]);
    freeArray(&m_dSeed[1]);

    freeArray(&m_dSortedPos);
    freeArray(&m_dSortedSeed);
    
    freeArray(&m_dSpinHash[0]);
    freeArray(&m_dSpinHash[1]);
 
    freeArray(&m_dFiberPos);
    freeArray(&m_dCubeCounter);
    freeArray(&m_dCubes);
    if(m_useVbo){
        unregisterGLBufferObject(m_posVbo[0]);
        unregisterGLBufferObject(m_posVbo[1]);
        glDeleteBuffers(2, m_posVbo);
        glDeleteBuffers(1, &m_colorVBO);
    }
}

/*
 *  Function: update()
 *  Return Type: void
 *  Parameters: float deltaTime
 *  Description: This function is the entry point for the spin position/movement computation 
 *               invokes the integrateSystem()
 * 
 */
float SpinSystem::update(float deltaTime, uint iterations){

	assert(m_bInitialized);
	float phaseConstant;
	float intraAdcScaled;
	float extraAdcScaled;
	float myelinAdcScaled;
	float s;
	float3 gradScaled;
	float kernelTime;

	
	// CONVERT UNITS

	// Convert mT/m to T/um (*1e-9) and then scale to our -1 to 1 space with  "* m_spaceScale"
	s = 1e-9f * m_spaceScale;
	gradScaled = m_gradient * s;

	// adc is passed as um^2/msec. deltaTime is in msec, so we just need to scale the
	// space to match our scaled space.
	//float myelinAdcScale = 0.01;

	extraAdcScaled = m_extraAdc/(m_spaceScale*m_spaceScale);
	intraAdcScaled = m_intraAdc/(m_spaceScale*m_spaceScale);
	myelinAdcScaled = m_myelinAdc/(m_spaceScale*m_spaceScale);

	//printf("m_extraAdc: %g\n", m_extraAdc);
	//printf("m_intraAdc: %g\n", m_intraAdc);
	//printf("m_myelinAdc: %g\n", m_myelinAdc);
	//printf("extraAdcScaled: %g\n", extraAdcScaled);
	//printf("intraAdcScaled: %g\n", intraAdcScaled);
	//printf("myelinAdcScaled: %g\n", myelinAdcScaled);

	// The expected phase shift is dot(G,pos) * 2*pi * gyromagneticRatio * deltaTime.
	// The dot-product will be computed in the spin kernel for each unique position.
	// We compute the rest out here for efficiency.
	// The gyromagnetic ratio is in KHz/T, which is equivalent to (cycles/millisecond)/T
	// Since our time units are ms, this means that we don't need to scale.
	phaseConstant = TWOPI * m_gyromagneticRatio * deltaTime;

    // For some reason, the cutXXTimer code needs a GLUT window, so it segfaults when
    // the display is disabled. However, on my system, the ANSI process timer ('clock') 
    // seems to give similar results to cutXXTimer, so we'll just use that.
    CUT_SAFE_CALL(cutResetTimer(m_timer));
    CUT_SAFE_CALL(cutStartTimer(m_timer)); 

	//printf("In SpinSystem::updateSpins: m_hPos[100]: %g\n", m_hPos[100]);
    //time_t start = clock();
	if(m_useGpu){
        	if(m_useVbo){
            	integrateSystemVbo( m_posVbo[m_currentPosRead],
                	                m_dSeed[m_currentSeedRead], 
                	                deltaTime,
                	                m_dFiberPos,
                	                m_permeability,
                	                intraAdcScaled,
                	                extraAdcScaled,
					myelinAdcScaled,
                	                m_numSpins,
                	                gradScaled,
                	                phaseConstant,
                	                m_dCubeCounter,
                	                m_dCubes,
                	                m_cubeLength,
                	                m_numCubes,
                	                m_maxFibersPerCube,
                	                m_innerRadiusProportion,
                	                iterations );
        	}else{
            	integrateSystem(    m_dPos[m_currentPosRead],
                	                m_dSeed[m_currentSeedRead], 
                	                deltaTime,
                	                m_dFiberPos,
                	                m_permeability,
                	                intraAdcScaled,
                	                extraAdcScaled,
					myelinAdcScaled,
                	                m_numSpins,
                	                gradScaled,
                	                phaseConstant,
                	                m_dCubeCounter,
                	                m_dCubes,
                	                m_cubeLength,
                	                m_numCubes,
                	                m_maxFibersPerCube,
                	                m_innerRadiusProportion,
                	                iterations );
        	}
	}else{
	   for(uint i=0; i<iterations; i++){

            	cpuIntegrateSystem( deltaTime, intraAdcScaled, extraAdcScaled, myelinAdcScaled, gradScaled, phaseConstant);

	   }
	}

	//printf("In SpinSystem::updateSpins: m_hPos[100]: %g\n", m_hPos[100]);

    CUT_SAFE_CALL(cutStopTimer(m_timer));
    kernelTime = cutGetTimerValue(m_timer)/1000.0f;

    //kernelTime = (float)(clock()-start)/CLOCKS_PER_SEC;
	m_totalNumTimeSteps += iterations;
    return kernelTime;
}

void SpinSystem::dumpGrid(){
    // debug
    copyArrayFromDevice(m_hCubeCounter, m_dCubeCounter, 0, sizeof(uint)*m_totalNumCubes);
    copyArrayFromDevice(m_hCubes, m_dCubes, 0, sizeof(uint)*m_totalNumCubes*m_maxFibersPerCube);
    uint total = 0;
    uint maxPerCell = 0;
    for(uint i=0; i<m_totalNumCubes; i++) {
        if (m_hCubeCounter[i] > maxPerCell)
            maxPerCell = m_hCubeCounter[i];
        if (m_hCubeCounter[i] > 0) {
            printf("%d (%d): ", i, m_hCubeCounter[i]);
            for(uint j=0; j< m_hCubeCounter[i]; j++) {
                printf("%d ", m_hCubes[i*m_maxFibersPerCube + j]);
            }
            total += m_hCubeCounter[i];
            printf("\n");
        }
    }
    printf("max per cell = %d\n", maxPerCell);
    printf("total = %d\n", total);
}

void SpinSystem::dumpSpins(uint start, uint count){   
    uint toRemoveWrning; 
    toRemoveWrning = start + count;
    double mrSignal = getMrSignal();
    printf("mrSignal = %g\n", mrSignal); 
}

double SpinSystem::getMrSignal(){
    //getArray();

    //
    // The mean MR signal (ignoring T1 and T2 effects) is simply (in matlab):
    //    abs(sum(exp(i*phase)))/numSpins
    // 
    // Doing this without complex math uses the relation exp(i*phase) = cos(phase) + i*sin(phase)
    // Also note that the complex conjugate (abs) in sin/cos form is sqrt(cos(phase)^2 + sin(phase)^2)
    // 
    double xMagn, yMagn, mrSignal;
    xMagn = 0; 
    yMagn = 0;
    // posPtr[0-2] is X,Y,Z; posPtr[3] used to be phase, is now not used
    float *posPtr = m_hPos;
    float *paramPtr = m_hPartParams;
    // Only measure spins in the center to avoid edge effects. d specifies
    // the proportion of the voxel to measure. E.g., d = 0.8 will measure all
    // spins that are within +/-0.8 from the center of the (-1 to +1) voxel.
    double d = 0.80f;
    uint n = 0;
	//printf("in mrGetSignal: m_hPos[100] = %g\n", m_hPos[100]);
    for(uint i=0; i<m_numSpins; i++) {
        if(posPtr[0]>-d && posPtr[0]<d && posPtr[1]>-d && posPtr[1]<d && posPtr[2]>-d && posPtr[2]<d){
           xMagn += cos(paramPtr[1])*paramPtr[0];
           yMagn += sin(paramPtr[1])*paramPtr[0];
           n++;
        }
        posPtr += 4;
	paramPtr += m_nParams;
    }
    mrSignal = sqrt(xMagn*xMagn+yMagn*yMagn)/(double)n;
    //printf("MR Signal: %g \n", mrSignal);
    return(mrSignal);
}

// Test

float* SpinSystem::getArray(){
    assert(m_bInitialized);
    float* hdata = m_hPos;
    float *ddata = m_dPos[m_currentPosRead];
    //printf("Getarray - m_currentPosRead:%u\n",m_currentPosRead);
    // Get the position array from the GPU to the host machine
    if(m_useVbo){
        unsigned int vbo = m_posVbo[m_currentPosRead];
        copyArrayFromDevice(hdata, ddata, vbo, m_numSpins*4*sizeof(float));
    }else{
        copyArrayFromDevice(hdata, ddata,0, m_numSpins*4*sizeof(float));
    }

    return hdata;
}

void SpinSystem::setArray(){
   // Copy the position array from the host machine to the GPU
    assert(m_bInitialized);
    if(m_useVbo){
        unregisterGLBufferObject(m_posVbo[m_currentPosRead]);
        glBindBuffer(GL_ARRAY_BUFFER, m_posVbo[m_currentPosRead]);
        glBufferSubData(GL_ARRAY_BUFFER, 0, m_numSpins*4*sizeof(float), m_hPos);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        registerGLBufferObject(m_posVbo[m_currentPosRead]);
	//printf("(Inside SpinSystem::setArray()): m_currentPosRead: %u, m_posVbo[m_currentPosRead]: %u\n", m_currentPosRead, m_posVbo[m_currentPosRead]);
    }else{
        copyArrayToDevice(m_dPos[m_currentPosRead], m_hPos,0, m_numSpins*4*sizeof(float));
    }    
    // set the seed array
    // (Copy the seed array from the host machine to the GPU)
    copyArrayToDevice(m_dSeed[m_currentSeedRead], m_hSeed, 0, m_numSpins*4*sizeof(uint));
}

void SpinSystem::initGrid(uint *size, float spacing, float jitter, uint numSpins){
    float offset[3] = {size[0]*spacing/2, size[1]*spacing/2, size[2]*spacing/2};
    for(uint z=0; z<size[2]; z++) {
        for(uint y=0; y<size[1]; y++) {
            for(uint x=0; x<size[0]; x++) {
                uint i = (z*size[1]*size[0]) + (y*size[0]) + x;
                if (i < numSpins) {//a different setup of the spin environment
                    m_hPos[i*4  ] = (spacing * x) + m_spinRadius - offset[0] + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+1] = (spacing * y) + m_spinRadius - offset[1] + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+2] = (spacing * z) + m_spinRadius - offset[2] + (frand()*2.0f-1.0f)*jitter;
		    m_hPartParams[i*m_nParams] = 1.0f; //magnitude
                    m_hPartParams[i*m_nParams+1] = 0.0f; //phase
                }
            }
        }
    }
}

void SpinSystem::resetPhase(){
 	for(uint i=0;i<m_numSpins;i++){
     	m_hPartParams[i*m_nParams+1] = 0.0f; //phase
	}
   setArray();
}

/*void SpinSystem::resetMagnitude(){							// Change 1/27: Added code
 	for(uint i=0;i<m_numSpins;i++){							//
      	m_hPartParams[i*m_nParams] = 1.0f; //magnitude				//
	}										//
   setArray();										//
}*/


void SpinSystem::reset(SpinConfig config){
	switch(config){
	default:
	case CONFIG_RANDOM:
		{
			//int p = 0;//, v = 0;
			for(uint i=0; i < m_numSpins; i++) 
			{
				float point[3];
				point[0] = frand();
				point[1] = frand();
				point[2] = frand();
				m_hPos[i*4] = 2 * (point[0] - 0.5f);
				m_hPos[i*4+1] = 2 * (point[1] - 0.5f);
				m_hPos[i*4+2] = 2 * (point[2] - 0.5f);
				m_hPartParams[i*m_nParams] = 1.0f; // magnitude
				m_hPartParams[i*m_nParams+1] = 0.0f; // phase
			}
		}
		break;

    case CONFIG_GRID:
        {
            float jitter = m_spinRadius*0.01f;
            uint s = (int) ceilf(powf((float) m_numSpins, 1.0f / 3.0f));
            uint numCubes[3];
            numCubes[0] = numCubes[1] = numCubes[2] = s;
            initGrid(numCubes, m_spinRadius*2.0f, jitter, m_numSpins);
        }
        break;
	}

    // Assign the random seed array (also stores the nearest fiber for each spin)
    assignSeeds();
    
    setArray();
}

void SpinSystem::colorSphere(float *pos, float radius, float *color){
    getArray(); 
    // set color buffer
    glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
    float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    float *ptr = data;
    for(uint i=0; i<m_numSpins; i++) {
       float dist = sqrt( (m_hPos[i*4  ]-pos[0])*(m_hPos[i*4  ]-pos[0])
                         +(m_hPos[i*4+1]-pos[1])*(m_hPos[i*4+1]-pos[1])
                         +(m_hPos[i*4+2]-pos[2])*(m_hPos[i*4+2]-pos[2]) );
       if(dist<radius){
          *ptr++ = color[0];
          *ptr++ = color[1];
          *ptr++ = color[2];
          *ptr++ = color[3];
       }else ptr+=4;
    }
    glUnmapBufferARB(GL_ARRAY_BUFFER);
}


void SpinSystem::setColorFromSpin(){
    getArray(); 
    // set color buffer
    glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
    float *colPtr = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    float phase;
    float *spinPtr = m_hPos+3;
    for(uint i=0; i<m_numSpins; i++) {
          // The phase angle (in radians) is stored in the 4th component if the position array.
          // Get the phase angle and set the color of this particle based on that.
          phase = sin(*spinPtr)/2;
          spinPtr+=5;
          *colPtr++ = 0.5 + phase;
          *colPtr++ = 0.5;
          *colPtr++ = 0.5;
          colPtr++;
    }
    glUnmapBufferARB(GL_ARRAY_BUFFER);
}


void SpinSystem::assignSeeds(){
	//uint insideIndex;
	//uint nearestFiber;
	uint2 nearestInfo;

   int numInside=0;
 	for(uint i=0;i<m_numSpins;i++){
 	   m_hSeed[i*4] = rand()*(UINT_MAX/RAND_MAX);
 	   m_hSeed[i*4+1] = rand()*(UINT_MAX/RAND_MAX);

	   nearestInfo = findNearestFiber(i);
	   m_hPartParams[i*m_nParams + 2] = (float) nearestInfo.a;
	   m_hPartParams[i*m_nParams + 3] = (float) nearestInfo.b;
	   if(nearestInfo.b < 2){
	      m_hSeed[i*4+3] = nearestInfo.a;
	      numInside++;
	   }else{
	      m_hSeed[i*4+3] = UINT_MAX;
	   }
	}
	printf("Num inside: %u \n", numInside);
	//cin.get();
}


/*
int SpinSystem::findNearestFiber(int index){
   //int insideFiberIndex = -1;
   //if((uint)index>=m_numSpins)
   //   return(insideFiberIndex);

   int insideFiberIndex = 0;
   
   int nearestTubeIndex = -1;
   
   float3 pos; 
	pos.x = m_hPos[index*4];
	pos.y = m_hPos[index*4+1];
	pos.z = m_hPos[index*4+2];
 
   float curDistToCenter;
   float3 tempPos = pos;

   uint cubeIndex;
   float3 cubePos;

   float minDistSq = 9.9e9;
   for(int y=-1;y<=1;y++){
      tempPos.y += y*m_cubeLength;
      if(fabs(tempPos.y)<=1){
         for(int x=-1;x<=1;x++){
            tempPos.x += x*m_cubeLength;
            if(fabs(tempPos.x)<=1){
               for(int z=-1;z<=1;z++){ 
                  tempPos.z += z*m_cubeLength;
                  if(fabs(tempPos.z)<=1){
                     cubePos = calcCubePos(tempPos);
                     cubeIndex = calcCubeHash(cubePos);   		      
                     for(uint j=0;j<m_hCubeCounter[cubeIndex];j++){
                        float3 fiberPos;
                        int i = m_hCubes[cubeIndex* m_maxFibersPerCube+j];
                        fiberPos.x = m_fiberPos[i*4];
                        fiberPos.y = m_fiberPos[i*4+1];
                        fiberPos.z = m_fiberPos[i*4+2];
                        float radSq = m_fiberPos[i*4+3];
                        // fiberPos.x|y|z == 2 means this is the longitudial axis of the fiber
                        if(fiberPos.x==2.0f){
                           curDistToCenter = (pos.z-fiberPos.z)*(pos.z-fiberPos.z) + (pos.y-fiberPos.y)*(pos.y-fiberPos.y);
                        }else if(fiberPos.z==2.0f){
                           curDistToCenter = (pos.x-fiberPos.x)*(pos.x-fiberPos.x) + (pos.y-fiberPos.y)*(pos.y-fiberPos.y);
                        }else{
                           curDistToCenter = (pos.x-fiberPos.x)*(pos.x-fiberPos.x) + (pos.z-fiberPos.z)*(pos.z-fiberPos.z);
                        }
                        if(curDistToCenter<0.64*radSq){
                        	insideFiberIndex = i;
                        }else if(curDistToCenter<radSq){
				insideFiberIndex = -i;
			}
                        if(curDistToCenter<minDistSq){
                           nearestTubeIndex = i;
                           minDistSq = curDistToCenter;
                        }
                     } tempPos.z -= z*m_cubeLength;
                  }
               }tempPos.x -= x*m_cubeLength;
            }
         }tempPos.y -= y*m_cubeLength;
      }
   }
   return(insideFiberIndex);
}
*/


uint2 SpinSystem::findNearestFiber(int index){
   uint imo = 2;						// Inside fiber/Myelin/Outside fiber (0/1/2)
   uint2 nearestInfo;
   float inRadPropSq = m_innerRadiusProportion*m_innerRadiusProportion;

   nearestInfo.a = 0;
   nearestInfo.b = 0;

   if((uint)index>=m_numSpins)
      return(nearestInfo);
   
   int nearestTubeIndex = -1;
   
   float3 pos; 
	pos.x = m_hPos[index*4];
	pos.y = m_hPos[index*4+1];
	pos.z = m_hPos[index*4+2];
 
   float curDistToCenter;
   float3 tempPos = pos;

   uint cubeIndex;
   float3 cubePos;


   float minDistSq = 9.9e9;
   for(int dy=-1;dy<=1;dy++){					// We check all fibers within the block of 3x3x3 cubes with the particle in the middle cube
      tempPos.y = pos.y + dy*m_cubeLength;
      if(fabs(tempPos.y)<=1){
         for(int dx=-1;dx<=1;dx++){
            tempPos.x = pos.x + dx*m_cubeLength;
            if(fabs(tempPos.x)<=1){
               for(int dz=-1;dz<=1;dz++){ 
                  tempPos.z = pos.z + dz*m_cubeLength;
                  if(fabs(tempPos.z)<=1){			// Now we are looking at one of the 27 cubes. We look at all fibers within that cube.

                     cubePos = calcCubePos(tempPos);
                     cubeIndex = calcCubeHash(cubePos);   					// cubeIndex is number of cube      
                     for(uint j=0;j<m_hCubeCounter[cubeIndex];j++){				// m_hCubeCounter[cubeIndex]: no. of fibers in cube no. cubeIndex
                        float3 fiberPos;
                        int i = m_hCubes[cubeIndex* m_maxFibersPerCube+j];			// i is number of fiber
                        fiberPos.x = m_fiberPos[i*4];
                        fiberPos.y = m_fiberPos[i*4+1];
                        fiberPos.z = m_fiberPos[i*4+2];
                        float radSq = m_fiberPos[i*4+3];
                        // fiberPos.x|y|z == 2 means this is the longitudinal axis of the fiber
                        if(fiberPos.x==2.0f){
                           curDistToCenter = (pos.z-fiberPos.z)*(pos.z-fiberPos.z) + (pos.y-fiberPos.y)*(pos.y-fiberPos.y);
                        }else if(fiberPos.z==2.0f){
                           curDistToCenter = (pos.x-fiberPos.x)*(pos.x-fiberPos.x) + (pos.y-fiberPos.y)*(pos.y-fiberPos.y);
                        }else{
                           curDistToCenter = (pos.x-fiberPos.x)*(pos.x-fiberPos.x) + (pos.z-fiberPos.z)*(pos.z-fiberPos.z);
                        }

			if (curDistToCenter<radSq){
		//		printf("Particle %u inside fiber %u!\n", index, i);
				if (curDistToCenter<inRadPropSq*radSq){
					imo = 0;
				} else {
					imo = 1;
				}
			}	// end (if curDistToCenter < radSq)


                        if(curDistToCenter<minDistSq){
                           nearestTubeIndex = i;
                           minDistSq = curDistToCenter;
                        }
                     } //tempPos.z -= z*m_cubeLength;		// end (for j)
                  }						// end (if fabs(z))
               }//tempPos.x -= x*m_cubeLength;
            }
         }//tempPos.y -= y*m_cubeLength;
      }
   }
	
//cin.get();

   nearestInfo.a = nearestTubeIndex;
   nearestInfo.b = imo;

//	if (curDistToCenter < radSq){
//		printf("*-* \n");
//		printf("Index: %g \n", index);
//		printf("CurdistToCenter \n")
//	}

   return(nearestInfo);
}



#include "spinKernelCpu.cpp"

