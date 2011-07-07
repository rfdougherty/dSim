//////////////////////////////////////////////////////////////////////////////////////////
// File name:		spinKernelCpu.cpp
// Description:		Kernel for spin computations using CPU
//////////////////////////////////////////////////////////////////////////////////////////

void SpinSystem::cpuIntegrateSystem(float deltaTime, float3 magneticGradient, float phaseConstant){

	//printf("
	
	for (uint i=0; i<m_numSpins; i++){
		cpuIntegrate(i, magneticGradient, phaseConstant, deltaTime);
	}	

	setArray();
	setSpinArray();
}


//////////////////////////////////////////////////////////////////////////////
// Function name:	boxMuller
// Description:		Generates a pair of independent standard normally
//			distributed random numbers from a pair of
//			uniformly distributed random numbers, using the basic form
//			of the Box-Muller transform 
//			(see http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)
//////////////////////////////////////////////////////////////////////////////
void SpinSystem::boxMuller(float& u1, float& u2){
	float r = sqrt(-2.0f * logf(u1));
	float phi = TWOPI * u2;
	u1 = r*cosf(phi);
	u2 = r*sinf(phi);
}


//////////////////////////////////////////////////////////////////////////////
// Function name:	myRand
// Description:		Simple multiply-with-carry PRNG that uses two seeds 
//			(seed[0] and seed[1]) (Algorithm from George Marsaglia: 
//			http://en.wikipedia.org/wiki/George_Marsaglia)
//////////////////////////////////////////////////////////////////////////////
uint SpinSystem::myRand(uint seed[]){
	seed[0] = 36969 * (seed[0] & 65535) + (seed[0] >> 16);
	seed[1] = 18000 * (seed[1] & 65535) + (seed[1] >> 16);
	return (seed[0] << 16) + seed[1];
}


//////////////////////////////////////////////////////////////////////////////
// Function name:	myRandn
// Description:		Returns three normally distributed random numbers 
//			and one from the uniformly distributed random number.
//////////////////////////////////////////////////////////////////////////////
void SpinSystem:: myRandn(uint seed[], float& n1, float& n2, float& n3, float& u) {
	// We want random numbers in the range (0,1]:
	n1 = ((float)myRand(seed) + 1.0f) / 4294967296.0f;
	n2 = ((float)myRand(seed) + 1.0f) / 4294967296.0f;
	n3 = ((float)myRand(seed) + 1.0f) / 4294967296.0f;
	u  = ((float)myRand(seed) + 1.0f) / 4294967296.0f;
	float n4 = u;
	boxMuller(n1, n2);
	boxMuller(n3, n4);
	return;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function name:	collDetect
// Description:		Determine whether a particle trying to travel from oPos to pos hits a triangle.
//			Use either the method of a rectangular grid or an R-Tree.
////////////////////////////////////////////////////////////////////////////////////////////////////////
float3 SpinSystem::collDetect(float3 oPos, float3 pos, float u, uint detectMethod, uint8 &compartment, uint16 &fiberInside){	
	
	if (detectMethod == 0){
		return collDetectRectGrid(oPos,pos,u,compartment,fiberInside);
	}
	else {
		return collDetectRTree(oPos,pos,u,compartment,fiberInside);
	}
}


//////////////////////////////////////////////////////////////////////////////////////////
// Function name:	collDetectRTree
// Description:		See whether a particle trying to go from oPos to pos
//			collides with any triangle in the mesh, using the R-Tree. Return
//			the final position of the particle.
//////////////////////////////////////////////////////////////////////////////////////////
float3 SpinSystem::collDetectRTree(float3 oPos, float3 pos, float u, uint8 &compartment, uint16 &fiberInside){
	
	float3 endPos = pos/*, collPoint*/;
	uint hitArray[100];				// Hitarray will store the indices of the triangles that the particle possible collides with - we are assuming no more than 100
	float minxyz[3], maxxyz[3];
	collResult result, tempResult;
	result.collDistSq = 400000000;			// Some really large number, will use this to store the smallest distance to a collision point
	result.collisionType = 1;
	result.collIndex = UINT_MAX;
	uint excludedTriangle = UINT_MAX;
	uint treeIndex;
	float u_max = 1, u_min = 0;

	// We start by picking the right R-Tree - each compartment of each fiber has its own R-Tree, and the space between the fiber has its
	// own tree as well, so pointers to the trees are stored in an array of size (m_numCompartments-1)*m_numFibers + 1, with first element representing
	// the space between fibers.
	if (compartment != 0){										// We push the location of the root node onto the stack
		treeIndex = fiberInside*(m_numCompartments-1)+compartment;	// = 0 for "first" tree, i.e. tree corresponding to innermost compartment
	} else{
		treeIndex = 0;
	}

	while (result.collisionType>0){			// If we have detected a collision, we repeat the collision detection for the new, reflected path
		result.collisionType = 0;		// First assume that the particle path does not experience any collisions
		
		// Define a rectangle that bounds the particle path from corner to corner
		// Finding minx, miny, minz
		minxyz[0] = oPos.x; if (pos.x < minxyz[0]){minxyz[0] = pos.x;}
		minxyz[1] = oPos.y; if (pos.y < minxyz[1]){minxyz[1] = pos.y;}
		minxyz[2] = oPos.z; if (pos.z < minxyz[2]){minxyz[2] = pos.z;}

		// Finding maxx, maxy, maxz
		maxxyz[0] = oPos.x; if (pos.x > maxxyz[0]){maxxyz[0] = pos.x;}
		maxxyz[1] = oPos.y; if (pos.y > maxxyz[1]){maxxyz[1] = pos.y;}
		maxxyz[2] = oPos.z; if (pos.z > maxxyz[2]){maxxyz[2] = pos.z;}

		// Find the triangles whose bounding rectangles intersect spinRectangle. They are written to hitArray and their number is nHits.
		int nHits = m_treeGroup[treeIndex].Search(minxyz, maxxyz, hitArray);

		// Loop through the triangles in hitArray, see if we have collisions, store the closest collision point in the variable result.
		for (uint k=0; k<nHits; k++){
			uint triIndex = hitArray[k];
			if (triIndex != excludedTriangle){
				tempResult = triCollDetect(oPos, pos, triIndex);
				if ((tempResult.collisionType>0) & (tempResult.collDistSq < result.collDistSq)){
					result = tempResult;
				}
			}
		}

		// If we have a collision, then we find the resulting point which the particle gets reflected to.
		if (result.collisionType>0){
			// If u>u_max-(u_max-u_min)*m_permeability, then the particle permeates through the membrane and does not get reflected.
			if (u<u_max-(u_max-u_min)*m_permeability){
				endPos = reflectPos(oPos, pos, result.collPoint, result.collIndex, result.collisionType);
				u_max = u_max-(u_max-u_min)*m_permeability;

				printf("Collision!\n");
				printf("oPos: [%g,%g,%g]\n",oPos.x,oPos.y,oPos.z);
				printf("pos: [%g,%g,%g]\n", pos.x,pos.y,pos.z);
				printf("collPoint: [%g,%g,%g]\n", result.collPoint.x,result.collPoint.y,result.collPoint.z);
				printf("endPos: [%g,%g,%g]\n", endPos.x, endPos.y, endPos.z);
				printf("compartment: %u\n", compartment);
				printf("fiberInside: %u\n", fiberInside);
			} else {
				u_min = u_max-(u_max-u_min)*m_permeability;

				// Change the compartment assignment of the spin to the new compartment
				//uint membraneType = m_hTriInfo[result.collIndex*3+1];
				if (compartment == 2){
					if (m_hTriInfo[result.collIndex*3+1] == 0){		// We are going from compartment 2 through axon surface - new compartment is 1
						compartment = 1;
					} else {							// We are going from compartment 2 through myelin surface - new compartment is 0
						compartment = 0;
						fiberInside = UINT16_MAX;
					}
				} else if (compartment == 1){
					compartment = 2;						// We are going from compartment 1 through axon surface - new compartment is 2
				} else if (compartment == 3){
					compartment = 0;
					fiberInside = UINT16_MAX;
				} else {
					fiberInside = m_hTriInfo[result.collIndex*3+0];
					if (m_hTriInfo[result.collIndex*3+1] == 1){		// We are going from compartment 0 through myelin surface - new compartment is 2
						compartment = 2;
					} else {							// We are going from compartment 0 through glia surface - new compartment is 3
						compartment = 3;
					}
				}

			}
		}

		// Redefine the start and end points for the reflected path, then repeat until no collision is detected.
		oPos = result.collPoint;
		pos = endPos;
		excludedTriangle = result.collIndex;	
		result.collDistSq = 400000000;
	}

	return endPos;
}


///////////////////////////////////////////////////////////////////////////////////////////////
// Function name:	collDetectRectGrid
// Description:		Determine whether a particle trying to go from oPos to pos
//			collides with a triangle, using the method of a rectangular grid (as 
//			opposed	to an R-Tree)
///////////////////////////////////////////////////////////////////////////////////////////////
float3 SpinSystem::collDetectRectGrid(float3 oPos, float3 pos, float u, uint8 &compartment, uint16 &fiberInside){

	float3 endPos = pos;
	float3 collPoint;
	uint excludedTriangle = UINT_MAX, currCube;
	collResult collCheck;
	collCheck.collisionType = 1;
	uint3 currCubexyz, startCubexyz, endCubexyz;
	int3 cubeIncrement;
	float u_max = 1.0f, u_min = 0.0f;

	while (collCheck.collisionType > 0){
		
		
		startCubexyz = calcCubePos(oPos);
		endCubexyz = calcCubePos(pos);
		cubeIncrement.x = ( (endCubexyz.x>startCubexyz.x) - (endCubexyz.x<startCubexyz.x) );
		cubeIncrement.y = ( (endCubexyz.y>startCubexyz.y) - (endCubexyz.y<startCubexyz.y) );
		cubeIncrement.z = ( (endCubexyz.z>startCubexyz.z) - (endCubexyz.z<startCubexyz.z) );


		collCheck.collisionType = 0;

		currCubexyz.x = startCubexyz.x;
		do {
			currCubexyz.y = startCubexyz.y;
			do {
				currCubexyz.z = startCubexyz.z;
				do {
					currCube = calcCubeHash(currCubexyz);
					//printf("currCubexyz: [%u,%u,%u]\n", currCubexyz.x, currCubexyz.y, currCubexyz.z);
					collCheck = cubeCollDetect(oPos, pos, currCube, excludedTriangle);
					currCubexyz.z += cubeIncrement.z;
				} while ((currCubexyz.z != endCubexyz.z+cubeIncrement.z)&&(collCheck.collisionType == 0));
				currCubexyz.y += cubeIncrement.y;
			} while ((currCubexyz.y != endCubexyz.y+cubeIncrement.y)&&(collCheck.collisionType == 0));
			currCubexyz.x += cubeIncrement.x;
		} while ((currCubexyz.x != endCubexyz.x+cubeIncrement.x)&&(collCheck.collisionType == 0));


		if (collCheck.collisionType > 0){

			if (u<=u_max-(u_max-u_min)*m_permeability){		// The spin does not permeate the membrane
				endPos = reflectPos(oPos, pos, collCheck.collPoint, collCheck.collIndex, collCheck.collisionType);
				u_max = u_max-(u_max-u_min)*m_permeability;
				//printf("(in spinKernel.cu::collDetectRTree): Particle bounces off membrane\n");
				//printf("(in spinKernel.cu::collDetectRTree): Endpos: [%g,%g,%g]\n", endPos.x, endPos.y, endPos.z);
				//reflectPos(oPos, pos, collCheck.collPoint, collCheck.collIndex, collCheck.collisionType);
			} else{							// The spin permeates the membrane
				u_min = u_max-(u_max-u_min)*m_permeability;

				// Change the compartment (and fiber, if appropriate) assignment of the spin
				// uint membraneType = tex1Dfetch(texTriInfo, collCheck.collIndex*3+1);
				if (compartment == 2){
					if (m_hTriInfo[collCheck.collIndex*3+1] == 0){		// We are going from compartment 2 through axon surface - new compartment is 1
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
					fiberInside = m_hTriInfo[collCheck.collIndex*3+0];
					if (m_hTriInfo[collCheck.collIndex*3+1] == 1){		// We are going from compartment 0 through myelin surface - new compartment is 2
						compartment = 2;
					} else {							// We are going from compartment 0 through glia surface - new compartment is 3
						compartment = 3;
					}
				}
			}
		}


		oPos = collCheck.collPoint;
		pos = endPos;
		excludedTriangle = collCheck.collIndex;	


		/*startCube = calcCubeHash(calcCubePos(oPos));				// The cube that the particle starts in
		endCube = calcCubeHash(calcCubePos(pos));				// The cube that the particle tries to end in

		collCheck = cubeCollDetect(oPos, pos, startCube, excludedTriangle);		// Find if the particle collides with any triangle in startCube

		if (collCheck.collisionType>0){						// If collision detected in startCube, find the position the particle bounces to
			endPos = reflectPos(oPos, pos, collCheck.collPoint, collCheck.collIndex, collCheck.collisionType, m_reflectionType);
		} else if (endCube != startCube){					// If no collision detected in startCube, do the same for endCube
			collCheck = cubeCollDetect(oPos, pos, endCube, excludedTriangle);
			if (collCheck.collisionType>0){
				endPos = reflectPos(oPos, pos, collCheck.collPoint, collCheck.collIndex, collCheck.collisionType, m_reflectionType);
			}
		}

		// Redefine the start and end points for the reflected path, then repeat until no collision is detected.
		oPos = collCheck.collPoint;
		pos = endPos;
		excludedTriangle = collCheck.collIndex;	*/				// Make sure we don't detect a collision with the triangle which the particle bounces from
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
collResult SpinSystem::cubeCollDetect(float3 oPos, float3 pos, uint cubeIndex, uint excludedTriangle){

	uint i, k_max;
	collResult result, tempResult;
	result.collisionType = 0;
	result.collIndex = UINT_MAX;
	result.collDistSq = 400000000;

	// Loop through membrane types (layers) as appropriate
	//for (uint layerIndex = 0; layerIndex < 2; layerIndex++){				// Change later so not to loop through all membrane types
		//k_max = m_hCubeCounter[layerIndex*m_totalNumCubes+cubeIndex];			// k_max: the number of triangles in cube cubeIndex on membrane type layerIndex
		k_max = m_hCubeCounter[cubeIndex];
		for (uint k=0; k<k_max; k++){							
			// i is the number of the triangle being checked.
			i = m_hTrianglesInCubes[cubeIndex*m_maxTrianglesPerCube+k];
			if (i != excludedTriangle){
				tempResult = triCollDetect(oPos, pos, i);

				if ((tempResult.collisionType>0) & (tempResult.collDistSq < result.collDistSq)){
					result = tempResult;
				}
			}
		}
	//}

	return result;
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
collResult SpinSystem::triCollDetect(float3 oPos, float3 pos, uint triIndex){
	
	float uv, wv, vv, wu, uu, r, s, t;
	collResult result;
	result.collIndex = UINT_MAX;
	result.collisionType = 0;
	result.collDistSq = 400000000;
	float3 triP1;
	uint P1 = m_hTriInfo[triIndex*3+2];
	triP1.x = m_vertices[P1*3+0];
	triP1.y = m_vertices[P1*3+1];
	triP1.z = m_vertices[P1*3+2];


	// First find whether the path intersects the plane defined by triangle i. See method at http://softsurfer.com/Archive/algorithm_0105/algorithm_0105.htm
	r = ( m_triangleHelpers[triIndex*12+0]*(triP1.x-oPos.x)+m_triangleHelpers[triIndex*12+1]*(triP1.y-oPos.y)+m_triangleHelpers[triIndex*12+2]*(triP1.z-oPos.z) ) / ( m_triangleHelpers[triIndex*12+0]*(pos.x-oPos.x)+m_triangleHelpers[triIndex*12+1]*(pos.y-oPos.y)+m_triangleHelpers[triIndex*12+2]*(pos.z-oPos.z) );
		
	if ((0<r)&(r<1)){
		// Then find if the path intersects the triangle itself. See method at http://softsurfer.com/Archive/algorithm_0105/algorithm_0105.htm
		wu = (oPos.x+r*(pos.x-oPos.x)-triP1.x)*m_triangleHelpers[triIndex*12+3]+(oPos.y+r*(pos.y-oPos.y)-triP1.y)*m_triangleHelpers[triIndex*12+4]+(oPos.z+r*(pos.z-oPos.z)-triP1.z)*m_triangleHelpers[triIndex*12+5];
		wv = (oPos.x+r*(pos.x-oPos.x)-triP1.x)*m_triangleHelpers[triIndex*12+6]+(oPos.y+r*(pos.y-oPos.y)-triP1.y)*m_triangleHelpers[triIndex*12+7]+(oPos.z+r*(pos.z-oPos.z)-triP1.z)*m_triangleHelpers[triIndex*12+8];

		s = (m_triangleHelpers[triIndex*12+9]*wv-m_triangleHelpers[triIndex*12+11]*wu)/(m_triangleHelpers[triIndex*12+9]*m_triangleHelpers[triIndex*12+9]-m_triangleHelpers[triIndex*12+10]*m_triangleHelpers[triIndex*12+11]);
		t = (m_triangleHelpers[triIndex*12+9]*wu-m_triangleHelpers[triIndex*12+10]*wv)/(m_triangleHelpers[triIndex*12+9]*m_triangleHelpers[triIndex*12+9]-m_triangleHelpers[triIndex*12+10]*m_triangleHelpers[triIndex*12+11]);

		if ( (s>=0)&(t>=0)&(s+t<=1) ){							// We have a collision with the triangle
			result.collPoint = oPos + r*(pos-oPos);
			result.collDistSq = (oPos.x-result.collPoint.x)*(oPos.x-result.collPoint.x)+(oPos.y-result.collPoint.y)*(oPos.y-result.collPoint.y)+(oPos.z-result.collPoint.z)*(oPos.z-result.collPoint.z);
			result.collIndex = triIndex;
			result.collisionType = 1;
			
			if ( (s==0)|(t==0)|(s+t==1) ){						// The collision point is on a triangle edge
				result.collisionType = 2;
				
				if ( ((s==0)&(t==0))|((s==0)&(t==1))|((s==1)&(t==0))){		// The collision point is on a triangle vertex
					result.collisionType = 3;
				}
			}
		}
	}
	return result;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function name:	reflectPos
// Description:		Given a particle that tries to travel from startPos to targetPos, but collides with triangle
//			number collTriIndex at collPos, we calculate the position which the particle gets reflected to.
//				This applies if reflectionType==1. If reflectionType==0, we do a simplified reflection,
//			where the particle just gets reflected to its original position. This is also done if we hit
//			a triangle edge or a triangle vertex (which gives collisionType==2 or collisionTYpe==3).
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float3 SpinSystem::reflectPos(float3 startPos, float3 targetPos, float3 collPos, uint collTriIndex, uint collisionType){

	float3 reflectedPos;

	if ((m_reflectionType==0)|(collisionType>1)){			// We simply reflect back to the starting point
			reflectedPos = startPos;
	} else {				// We reflect the target point through the triangle - see http://en.wikipedia.org/wiki/Transformation_matrix
			float3 sPosShifted = targetPos-collPos;
			float3 normalVec;
			normalVec = make_float3(m_triangleHelpers[collTriIndex*12+0],m_triangleHelpers[collTriIndex*12+1],m_triangleHelpers[collTriIndex*12+2]);
			reflectedPos.x = (1-2*normalVec.x*normalVec.x)*sPosShifted.x - 2*normalVec.x*normalVec.y*sPosShifted.y - 2*normalVec.x*normalVec.z*sPosShifted.z + collPos.x;
			reflectedPos.y = -2*normalVec.x*normalVec.y*sPosShifted.x + (1-2*normalVec.y*normalVec.y)*sPosShifted.y - 2*normalVec.y*normalVec.z*sPosShifted.z + collPos.y;
			reflectedPos.z = -2*normalVec.x*normalVec.z*sPosShifted.x - 2*normalVec.y*normalVec.z*sPosShifted.y + (1-2*normalVec.z*normalVec.z)*sPosShifted.z + collPos.z;
	}

	return reflectedPos;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
// Function name:	cpuIntegrate
// Description:		"Main" function for CPU kernel computation, called from spinSystem.cpp. 
//			Computes the spin movement and signal for each spin.
///////////////////////////////////////////////////////////////////////////////////////////////////////
void SpinSystem::cpuIntegrate(int spinIndex, float3 magneticGradient, float phaseConstant,/* float intraStdDev, float extraStdDev, float myelinStdDev, float gliaStdDev, */float deltaTime/*, float radPrSq*/){

	float3 pos;
	pos.x = m_hPos[spinIndex*m_nPosValues];
	pos.y = m_hPos[spinIndex*m_nPosValues+1];
	pos.z = m_hPos[spinIndex*m_nPosValues+2];
	float3 oPos = pos;							// Store a copy of the old position before we update it

	//if (spinIndex == 0){
	//	printf("[%g,%g,%g]\n", pos.x, pos.y, pos.z);
	//	printf("Signal magnitude: %g\n", m_hSpins[spinIndex].signalMagnitude);
	//	printf("m_hT2Values: [%g,%g,%g,%g]\n", m_hT2Values[0], m_hT2Values[1], m_hT2Values[2], m_hT2Values[3]);
	//	printf("deltaTime: %g\n", deltaTime);
	//}


	uint seed[2];
	seed[0] = m_hSeed[m_nSeedValues*spinIndex];
	seed[1] = m_hSeed[m_nSeedValues*spinIndex+1];
	
	float3 brnMot;
	float u;
	myRandn(seed, brnMot.x, brnMot.y, brnMot.z, u);

	//uint compartment = m_hSpins[spinIndex*m_nSpinValues];
	//uint fiberInside = m_hSpins[spinIndex*m_nSpinValues+3];
	uint8 compartment = m_hSpins[spinIndex].compartmentType;
	uint16 fiberInside = m_hSpins[spinIndex].insideFiber;

	pos.x += brnMot.x * m_hStdDevs[compartment];
	pos.y += brnMot.y * m_hStdDevs[compartment];
	pos.z += brnMot.z * m_hStdDevs[compartment];	

	//pos += brnMot*m_hAdcValues[compartment];

	// Do a collision detection for the path the particle is trying to take
	pos = collDetect(oPos,pos,u,m_triSearchMethod,compartment,fiberInside);

	// Don't let the spin leave the volume
	if (pos.x >  1.0f - m_spinRadius) { pos.x =  1.0f - m_spinRadius; }
	else if (pos.x < -1.0f + m_spinRadius) { pos.x = -1.0f + m_spinRadius; }
	if (pos.y >  1.0f - m_spinRadius) { pos.y =  1.0f - m_spinRadius; }
	else if (pos.y < -1.0f + m_spinRadius) { pos.y = -1.0f + m_spinRadius; }
	if (pos.z >  1.0f - m_spinRadius) { pos.z =  1.0f - m_spinRadius; }
	else if (pos.z < -1.0f + m_spinRadius) { pos.z = -1.0f + m_spinRadius; }

	// Store new position
	m_hPos[spinIndex*m_nPosValues] = pos.x;
	m_hPos[spinIndex*m_nPosValues+1] = pos.y;
	m_hPos[spinIndex*m_nPosValues+2] = pos.z;
	
	// Update random seeds
	m_hSeed[spinIndex*m_nSeedValues] = seed[0];
	m_hSeed[spinIndex*m_nSeedValues+1] = seed[1];

	//m_hSpins[spinIndex*m_nSpinValues+1] += -m_hSpins[spinIndex*m_nSpinValues+1]/m_hT2Values[compartment]*deltaTime;
	m_hSpins[spinIndex].signalMagnitude += -m_hSpins[spinIndex].signalMagnitude/m_hT2Values[compartment]*deltaTime;

	// Update MR signal phase
	//m_hSpins[spinIndex*m_nSpinValues+2] += (magneticGradient.x * pos.x + magneticGradient.y * pos.y + magneticGradient.z * pos.z) * phaseConstant;
	m_hSpins[spinIndex].signalPhase += (magneticGradient.x * pos.x + magneticGradient.y * pos.y + magneticGradient.z * pos.z) * phaseConstant;

	// Update which fiber the spin belongs to
	//m_hSpins[spinIndex*m_nSpinValues+3] = fiberInside;
	m_hSpins[spinIndex].insideFiber = fiberInside;

	// Update the compartment type the spin is in
	//m_hSpins[spinIndex*m_nSpinValues+3] = fiberInside;
	m_hSpins[spinIndex].compartmentType = compartment;
}
