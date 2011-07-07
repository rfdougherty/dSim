
void 
SpinSystem::cpuIntegrateSystem(	float deltaTime,float intraAdc,float extraAdc, float myelinAdc, float3 magneticGradient,float phaseConstant)
{
	// To avoide extra computation in the kernel, we compute the random walk 
	// standard deviation out here and pass it in.

	float intraStdDev = sqrt(2.0f * intraAdc * deltaTime);
	float extraStdDev = sqrt(2.0f * extraAdc * deltaTime);
	float myelinStdDev = sqrt(2.0f * myelinAdc * deltaTime);

	float inRadPropSq = m_innerRadiusProportion*m_innerRadiusProportion;


 	for(uint i=0;i<m_numSpins;i++){
		cpuIntegrate(i,magneticGradient, phaseConstant, intraStdDev, extraStdDev, myelinStdDev, deltaTime, inRadPropSq);
	}
   setArray();

}


//////////////////////////////////////////////////////////////////////////////////////////
// The Box-Muller transform generates pairs of independent standard normally distributed
// random numbers, given a pair of uniformly distributed random numbers.
// The form used here is the basic form, as opposed to the polar form.
// See explanation at http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
//////////////////////////////////////////////////////////////////////////////////////////
void
SpinSystem::boxMuller(float& u1, float& u2){
    float   r = sqrt(-2.0f * logf(u1));
    float phi = TWOPI * u2;
    u1 = r * cosf(phi);
    u2 = r * sinf(phi);
}

uint 
SpinSystem::rand31pmc(uint &seed){
   uint hi, lo;
   lo = 16807 * (seed & 0xFFFF);
   hi = 16807 * (seed >> 16);
   lo += (hi & 0x7FFF) << 16;
   lo += hi >> 15;                  
   if (lo > 0x7FFFFFFF) lo -= 0x7FFFFFFF;          
   return ( seed = lo );        
}

uint 
SpinSystem:: myRand(uint seed[]){
   // Simple multiply-with-carry PRNG that uses two seeds (seed[0] and seed[1])
   // (Algorithm from George Marsaglia: http://en.wikipedia.org/wiki/George_Marsaglia)
    seed[0] = 36969 * (seed[0] & 65535) + (seed[0] >> 16);
    seed[1] = 18000 * (seed[1] & 65535) + (seed[1] >> 16);
    return (seed[0] << 16) + seed[1];
}

void 
SpinSystem:: myRandn(uint seed[], float& n1, float& n2, float& n3, float& u) {
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


float
SpinSystem:: distToCenterSq(float3 partPos, float3 fiberPos) {
	float dist;
	if (fiberPos.x == 2.0f){
		dist = (partPos.z-fiberPos.z)*(partPos.z-fiberPos.z) + (partPos.y-fiberPos.y)*(partPos.y-fiberPos.y);
	}else if (fiberPos.z == 2.0f){
		dist = (partPos.x-fiberPos.x)*(partPos.x-fiberPos.x) + (partPos.y-fiberPos.y)*(partPos.y-fiberPos.y);
	}else{
		dist = (partPos.x-fiberPos.x)*(partPos.x-fiberPos.x) + (partPos.z-fiberPos.z)*(partPos.z-fiberPos.z);
	}
	return dist;
}


///////////////////////////////////////////////////////////////////////////////////////
// This is a special kind of sign function which gives -1 if x < 0 and 1 otherwise
///////////////////////////////////////////////////////////////////////////////////////
int
SpinSystem::signum(float x){return (x >= 0) - (x < 0);}

////////////////////////////////////////////////////////////////////////////////////////
// reflectedPos determines the end position of the particle if it tries to enter/exit a 
// fiber but is reflected.
//	Inputs: startPos: Original position of particle
//		targetPos: Where particle would have ended up if no reflection
//		fiberPos: Position of fiber (coordinate which equals zero determines axis orientation
//		radSq: Square value of fiber radius
//
//	Output: Position of particle after reflection
////////////////////////////////////////////////////////////////////////////////////////
float3
SpinSystem::reflectedPos(float3 startPos, float3 targetPos, float3 fiberPos, float inputRadSq, uint imo, bool outerMembrane, float inRadPropSq) {

	//return startPos;	// temp test
	//printf("Running reflectedPos\n");
	float3 reflectPos = targetPos;
	float3 interPos;
	float refRadSq;


	////////////////////////////////////////////////////////////////////////////////
	// c1 and c2 stands for coordinate 1 and coordinate 2. This is to avoid confusion since
	// we are not always in the xy-plane (i.e. the fibers are not necessarily oriented along
	// the z-axis).
	// t, s, f, i and r stands for target, start, fiber, intersection and reflection, so for
	// example sc2 means coordinate 2 of the starting point of the particle. Intersection is
	// where the original particle path meets the fiber membrane.
	////////////////////////////////////////////////////////////////////////////////
	float tc1, tc2, sc1, sc2, fc1, fc2, dc1, dc2, dr, D, ic1, ic2, rc1, rc2;

	if(fiberPos.z==2.0f){		// Fiber oriented along z axis
		tc1 = targetPos.x;
		tc2 = targetPos.y;
		sc1 = startPos.x;
		sc2 = startPos.y;
		fc1 = fiberPos.x;
		fc2 = fiberPos.y;
	}else if(fiberPos.x==2.0f){	// Fiber oriented along x axis
		tc1 = targetPos.y;
		tc2 = targetPos.z;
		sc1 = startPos.y;
		sc2 = startPos.z;
		fc1 = fiberPos.y;
		fc2 = fiberPos.z;
	}else{				// Fiber oriented along y axis
		tc1 = targetPos.z;
		tc2 = targetPos.x;
		sc1 = startPos.z;
		sc2 = startPos.x;
		fc1 = fiberPos.z;
		fc2 = fiberPos.x;            
	}

	if (outerMembrane){
		refRadSq = inputRadSq;
	} else {
		refRadSq = inRadPropSq*inputRadSq;
	}
	//////////////////////////////////////////////////////////////////////////
	// First we find the point (ic1,ic2) where the particle hits the fiber.
	// The method is described at http://mathworld.wolfram.com/Circle-LineIntersection.html
	//////////////////////////////////////////////////////////////////////////
	dc1 = tc1 - sc1;
	dc2 = tc2 - sc2;
	dr = sqrt(dc1*dc1 + dc2*dc2);
	D = (sc1-fc1)*(tc2-fc2) - (tc1-fc1)*(sc2-fc2);

	/////////////////////////////////////////////////////////////////////////////////
	// ic2a and ic2b are two possible solutions for the c2-coordinate of the intersection point.
	// We calculate them both and check manually which one is correct (between the start point and the
	// target point). There might be an easier way to do this.
	// NB: Might give incorrect results when target and start have same c2-coordinate (dc2 = 0)
	/////////////////////////////////////////////////////////////////////////////////
	float ic2a = (-D*dc1 + fabs(dc2)*sqrt(refRadSq*dr*dr - D*D))/(dr*dr) + fc2;
	float ic2b = (-D*dc1 - fabs(dc2)*sqrt(refRadSq*dr*dr - D*D))/(dr*dr) + fc2;
	float ic1a = (D*dc2 + signum(dc2)*dc1*sqrt(refRadSq*dr*dr - D*D))/(dr*dr) + fc1;
	float ic1b = (D*dc2 - signum(dc2)*dc1*sqrt(refRadSq*dr*dr - D*D))/(dr*dr) + fc1;



	bool ainbetween = ( ((sc1>ic1a) & (ic1a>=tc1)) | ((tc1>=ic1a) & (ic1a>=tc1)) | ((sc2>ic2a) & (ic2a>=tc2)) | ((tc2>=ic2a) & (ic2a>sc2)) );
	bool binbetween = ( ((sc1>ic1b) & (ic1b>=tc1)) | ((tc1>=ic1b) & (ic1b>=tc1)) | ((sc2>ic2b) & (ic2b>=tc2)) | ((tc2>=ic2b) & (ic2b>sc2)) );

	if (ainbetween & !binbetween) {
		ic1 = ic1a;
		ic2 = ic2a;
	} else if (!ainbetween & binbetween) {
		ic1 = ic1b;
		ic2 = ic2b;
	} else if ( (ic1a-tc1)*(ic1a-tc1)+(ic2a-tc2)*(ic2a-tc2) < (ic1b-tc1)*(ic1b-tc1)+(ic2b-tc2)*(ic2b-tc2) ) {
		//printf("a closer to target\n");
		ic1 = ic1a;
		ic2 = ic2a;
	} else {
		//printf("b closer to target\n");
		ic1 = ic1b;
		ic2 = ic2b;
	}
	
	

	///////////////////////////////////////////////////////////////////////////
	// We then find the point (rc1,rc2) where the reflected particle ends up by reflecting the original
	// target point by the line going through (xi,yi) parallel to the fiber. A description
	// of the method can be found at http://mathworld.wolfram.com/Reflection.html
	///////////////////////////////////////////////////////////////////////////
	rc1 = 2*(-(tc1-ic1)*(ic2-fc2)+(tc2-ic2)*(ic1-fc1))/((ic2-fc2)*(ic2-fc2)+(ic1-fc1)*(ic1-fc1))*(fc2-ic2) + 2*ic1 - tc1;
	rc2 = 2*(-(tc1-ic1)*(ic2-fc2)+(tc2-ic2)*(ic1-fc1))/((ic2-fc2)*(ic2-fc2)+(ic1-fc1)*(ic1-fc1))*(ic1-fc1) + 2*ic2 - tc2;

	if(fiberPos.z==2.0f){		// Fiber oriented along z axis
		reflectPos.x = rc1;
		reflectPos.y = rc2;
		interPos.x = ic1;
		interPos.y = ic2;
		interPos.z = 0.0f;
	}else if(fiberPos.x==2.0f){	// Fiber oriented along x axis
		reflectPos.y = rc1;
		reflectPos.z = rc2;
		interPos.x = 0.0f;
		interPos.y = ic1;
		interPos.z = ic2;
	}else{				// Fiber oriented along y axis
		reflectPos.x = rc2;
		reflectPos.z = rc1;
		interPos.x = ic2;
		interPos.y = 0.0f;
		interPos.z = ic1;
	}
	


	//if ( (interPos.x == startPos.x) && (interPos.y == startPos.y) ) {
	//if ((ainbetween & binbetween) | (!ainbetween & !binbetween)) {
	/*	printf("*\nReflection!\n");
		printf("Start position, x,y,z: %.15g, %.15g, %g\n", startPos.x, startPos.y, startPos.z);
		printf("Target position, x,y,z: %.15g, %.15g, %g\n", targetPos.x, targetPos.y, targetPos.z);
		printf("Fiber position, x,y,z: %g, %g, %g\n", fiberPos.x, fiberPos.y, fiberPos.z);
		printf("Intersection position, x,y,z: %.15g, %.15g, %g\n", ic1, ic2, 0.0f);
		printf("Reflection radius squared: %g\n", refRadSq);
		printf("End position, x,y,z: %g, %g, %g\n", reflectPos.x, reflectPos.y, reflectPos.z);
		printf("imo: %u\n", imo);
		printf("Outermembrane: %u\n", outerMembrane);
		printf("ic1a: %g, ic2a: %g\n", ic1a, ic2a);
		printf("ic1b: %g, ic2b: %g\n", ic1b, ic2b);
		printf("dc1: %g, dc2: %g\n", dc1, dc2);
		printf("dr: %g, D: %g\n", dr, D);
		printf("Stuff in sqrt: %g\n*\n\n", refRadSq*dr*dr - D*D);*/
	

	///////////////////////////////////////////////////////////////////////////////////////////////////
	// If the particle starts in the myelin sheath, we check if it gets reflected out of the fiber or into
	// the fiber core. If it does, we check to see wether it gets reflected again. If it does, we apply the
	// reflection method recursively until the particle permeates a membrane or ends in the myelin after
	// an undetermined number of reflections between the inner and outer membrane.
	///////////////////////////////////////////////////////////////////////////////////////////////////
	float newDist = distToCenterSq(reflectPos,fiberPos);
	float p;	


	if  (imo == 1) {
		if (newDist > inputRadSq) {								// Particle wants to go through outer membrane
			p = frand();
			//printf("p: %g\n", p);
			if (p > m_permeability){							// It gets reflected again
				//printf("Another reflection of myelin particle - from outer membrane!\n");
				return reflectedPos(interPos,reflectPos,fiberPos,inputRadSq,imo, 1,inRadPropSq);
			} else {
				return reflectPos;
			}
		} else if (newDist < inRadPropSq*inputRadSq) {							// Particle wants to go through inner membrane
			p = frand();
			//printf("p: %g\n", p);
			if (p > m_permeability){							// It gets reflected again
				//printf("Another reflection of myelin particle - from inner membrane!\n");
				return reflectedPos(interPos,reflectPos,fiberPos,inputRadSq,imo, 0, inRadPropSq);
			} else {
				return reflectPos;
			}
		} else { 
			//printf("No more reflections...\n");
			return reflectPos;
		}
	} else if (imo == 0) {
		if (newDist > inputRadSq) {								// Particle wants to go from core to outside
			p = frand();
			//printf("p: %g\n", p);
			if (p < 1-m_permeability){							// It gets reflected from inner membrane
				//printf("Another reflection of core particle (trying to get outside) - from inner membrane!\n");
				return reflectedPos(interPos,reflectPos,fiberPos,inputRadSq,imo,0,inRadPropSq);
			}
			else if ((1-m_permeability < p) & (p < 1-m_permeability*m_permeability)) {	// It gets reflected from outer membrane
				//printf("Another reflection of core particle (trying to get outside) - from outer membrane!\n");
				return reflectedPos(interPos,reflectPos,fiberPos,inputRadSq,1,1,inRadPropSq);
			}
			else {
				return reflectPos;
			}
		}
		else if (newDist > inRadPropSq*inputRadSq) {
			p = frand();
			//printf("p: %g\n", p);
			if (p > m_permeability){
				//printf("Another reflection of core particle (trying to get to myelin) - from inner membrane!\n");
				return reflectedPos(interPos,reflectPos,fiberPos,inputRadSq,imo,0,inRadPropSq);
			}
			else {
				return reflectPos;
			}
		} else {
			// printf("No more reflections...\n");
			return reflectPos;
		}
	} else {
		return reflectPos;
	}

}




void 
SpinSystem::cpuIntegrate(int index,float3 magneticGradient,float phaseConstant,float intraStdDev,float extraStdDev, float myelinStdDev, float deltaTime, float inRadPropSq)
{

	//printf("Running cpuIntegrate in spinKernelCpu with index = %i\n", index);

	//if (index == 100){
	//	printf("At spin no. 100\n");
	//}

	if((uint)index>=m_numSpins)
	return;


	float3 pos; 
	pos.x = m_hPos[index*4];							// Change 1/28: Changed 4 to 5
	pos.y = m_hPos[index*4+1];							// Change 1/28: Changed 4 to 5
	pos.z = m_hPos[index*4+2];							// Change 1/28: Changed 4 to 5
	float3 oPos = pos;

	float3 vel;

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

	float3 brnMot;
	uint seed[2];
	seed[0] = m_hSeed[4*index];
	seed[1] = m_hSeed[4*index+1];
	//seed[1] = m_hSeed[index] + clock() + (uint)index;
	//uint insideFiberIndex = m_hSeed[4*index+3];
	uint nearestFiberIndex = m_hPartParams[index*m_nParams + 2];
	uint imo = m_hPartParams[index*m_nParams + 3];
	
	bool isInside = (imo<2);

	// Take a random walk...
	float u; // we get a bonus random number for free :)
	myRandn(seed, brnMot.x, brnMot.y, brnMot.z, u);

	if(imo == 0){
		vel.x = brnMot.x * intraStdDev;
		vel.y = brnMot.y * intraStdDev;
		vel.z = brnMot.z * intraStdDev;      
	}else if (imo == 1){
		vel.x = brnMot.x * myelinStdDev;
		vel.y = brnMot.y * myelinStdDev;
		vel.z = brnMot.z * myelinStdDev;
	}else {
		vel.x = brnMot.x * extraStdDev;
		vel.y = brnMot.y * extraStdDev;
		vel.z = brnMot.z * extraStdDev;
	}
	
	pos.x += vel.x;
	pos.y += vel.y;
	pos.z += vel.z;

	//printf("Index: %u, oldx: %g, oldy: %g, oldz:%g, newx: %g, newy: %g, newz:%g, nearestFiberIndex: %u, fx: %g, fy: %g, fz: %g, r2: %g, imo: %u \n", index, oPos.x, oPos.y, oPos.z, pos.x, pos.y, pos.z, nearestFiberIndex, m_fiberPos[nearestFiberIndex*4], m_fiberPos[nearestFiberIndex*4+1], m_fiberPos[nearestFiberIndex*4+2], m_fiberPos[nearestFiberIndex*4+3], imo);

	

	//printf("Checking if the particle permeates through a membrane. \n");
	bool reflected = false;
	if(m_permeability<1.0f){
		float radSq;
		float curDistToCenter;
		float oldDistToCenter;


		if (imo < 2){								// Particle is in myelin or in inner fiber
			//printf("Particle %u starts inside fiber, new x,y,z: %g, %g, %g \n", index, pos.x, pos.y, pos.z);
			//printf("imo: %u\n", imo);
			float3 fiberPos;
			fiberPos.x = m_fiberPos[nearestFiberIndex*4];
			fiberPos.y = m_fiberPos[nearestFiberIndex*4+1];
			fiberPos.z = m_fiberPos[nearestFiberIndex*4+2];
			radSq = m_fiberPos[nearestFiberIndex*4+3];

			oldDistToCenter = distToCenterSq(oPos,fiberPos);
			/*if ((imo == 0) & (oldDistToCenter > inRadPropSq*radSq)) {
				printf("Error: imo 0 but particle outside\n");
				printf("Index: %u\n", index);
				printf("fiberPos.x: %g, fiberPos.y: %g, radSq: %g\n", fiberPos.x, fiberPos.y, radSq);
				printf("oPos.x: %g, oPos.y: %g\n", oPos.x, oPos.y);
				cin.get();
			} else if ( (imo == 1) & ((oldDistToCenter < inRadPropSq*radSq) | (oldDistToCenter > radSq)) ) {
				printf("Error: imo 1 but particle not in myelin\n");
				printf("Index: %u\n", index);
				printf("fiberPos.x: %g, fiberPos.y: %g, radSq: %g\n", fiberPos.x, fiberPos.y, radSq);
				printf("oPos.x: %g, oPos.y: %g\n", oPos.x, oPos.y);
				cin.get();				
			}*/


			curDistToCenter = distToCenterSq(pos, fiberPos);
	
			if (imo == 0){									// Particle is in inner fiber
				if ( (curDistToCenter > inRadPropSq*radSq) & (curDistToCenter <= radSq) ){	// Particle tries to exit inner fiber to myelin
					if (u >= m_permeability) {					// Particle cannot escape from inner fiber
						//printf("Particle tries to get to myelin from core but gets reflected\n");
						reflected = true;					// Unnecessary line?
						//pos = reflectedPos(oPos,pos,fiberPos,radSq,imo,0,inRadPropSq);
						pos = oPos;
						//printf("pos.x: %g, pos.y: %g \n", pos.x, pos.y);
					}
					else{								// Particle exits inner fiber
						//printf("Particle goes from core to myelin\n");
						imo = 1;
					}
				} else if ( curDistToCenter > radSq ){					// Particle tries to exit inner fiber to outside
					if (u < 1-m_permeability) {					// Particle cannot exit inner fiber
						//printf("Particle tries to get outside from core but cannot escape core\n");
						reflected = true;
						//pos = reflectedPos(oPos,pos,fiberPos,radSq,imo,0,inRadPropSq);		
						pos = oPos;
						//printf("pos.x: %g, pos.y: %g \n", pos.x, pos.y);
					}
					else if ((1-m_permeability < u) & (u < 1-m_permeability*m_permeability)){	// Particle exits inner fiber but cannot exit myelin
						//printf("Particle goes from core to myelin but cannot reach outside\n");
						reflected = true;
						//pos = reflectedPos(oPos,pos,fiberPos,radSq,1,1,inRadPropSq);
						pos = oPos;
						imo = 1;
					}
					else {								// Particle reaches outside of fiber
						//printf("Particle goes from core to outside\n");
						imo = 2;
					}
				}
			} else {									// Particle is in myelin

				if (curDistToCenter > radSq){						// Particle tries to reach outside of fiber
					if (u >= m_permeability) {					// Particle is reflected back to myelin.
						//printf("Particle tries to go from myelin to outside but gets reflected\n");
						reflected = true;
						//pos = reflectedPos(oPos,pos,fiberPos,radSq,imo,1,inRadPropSq);
						pos = oPos;
					} else {		// Particle exits fiber
						//printf("Particle goes from myelin to outside\n");
						imo = 2;
					}
				} else if (curDistToCenter < inRadPropSq*radSq) {
					if (u >= m_permeability) {					// Particle tries to reach inner fiber
						reflected = true;					// Particle is reflected back to myelin.
						//printf("Particle tries to go from myelin to core but gets reflected\n");
						//pos = reflectedPos(oPos,pos,fiberPos,radSq,imo,0,inRadPropSq);
						pos = oPos;
					} else {		// Particle enters inner fiber
						//printf("Particle goes from myelin to core\n");
						imo = 0;
					}
				}
			}

		
			float endDist = distToCenterSq(pos,fiberPos);
			//printf("pos.x: %g, pos.y: %g, pos.z: %g, fiberPos.x: %g, fiberPos.y: %g, fiberPos.z: %g, radSq: %g \n", pos.x, pos.y, pos.z, fiberPos.x, fiberPos.y, fiberPos.z, radSq);
			if (endDist > radSq){
				imo = 2;
			//	printf("Ends outside fiber\n");
			} else if (endDist < inRadPropSq*radSq){
				imo = 0;
			//	printf("Ends in fiber core\n");
			} else{
				imo = 1;
			//	printf("Ends in myelin\n");
			}

			
		} else	{										// Particle is outside fiber
			//printf("Particle %u starts outside fiber, new x,y,z: %g, %g, %g \n", index, pos.x, pos.y, pos.z);
			//printf("imo: %u\n", imo);
			uint cubeIndex;
			float3 cubePos = calcCubePos(pos);
			cubeIndex = calcCubeHash(cubePos);  		      
			//printf("cp.x: %g, cp.y: %g, cp.z: %g, ci: %u, ", cubePos.x, cubePos.y, cubePos.z, cubeIndex);

			for(uint j=0;j<m_hCubeCounter[cubeIndex];j++){
				float3 fiberPos;
				int i = (m_hCubes[cubeIndex* m_maxFibersPerCube+j]);
				//if (j < m_hCubeCounter[cubeIndex] - 1) {printf("%u,", i);} else {printf("%u\n", i);}
				fiberPos.x = m_fiberPos[i*4];
				fiberPos.y = m_fiberPos[i*4+1];
				fiberPos.z = m_fiberPos[i*4+2];
				radSq = m_fiberPos[i*4+3];

				oldDistToCenter = distToCenterSq(oPos,fiberPos);
				/*if (oldDistToCenter < radSq) {
					printf("Error: imo 2 but particle inside\n");
					printf("Index: %u\n", index);
					printf("fiberIndex: %u, fiberPos.x: %g, fiberPos.y: %g, radSq: %g\n", i, fiberPos.x, fiberPos.y, radSq);
					printf("oPos.x: %g, oPos.y: %g\n", oPos.x, oPos.y);
					
					cin.get();
				}*/
            	
				curDistToCenter = distToCenterSq(pos,fiberPos);
				
				if ((curDistToCenter<=radSq) & (curDistToCenter > inRadPropSq*radSq)) {	// Particle tries to enter myelin from outside
					if(u>=m_permeability) {		// Particle cannot enter fiber
						//printf("Particle outside, target in myelin, reflected from outer fiber, curDistToCenter: %g, tube index: %u\n", curDistToCenter, i);
						//printf("Fiber pos: x: %g, y: %g, r2: %g\n", fiberPos.x, fiberPos.y, radSq);
						reflected = true;
						//pos = reflectedPos(oPos,pos,fiberPos,radSq,imo,1,inRadPropSq);
						pos = oPos;
						//printf("New position: x: %g, y: %g, z: %g\n", pos.x, pos.y, pos.z);
					}
					else {				// Particle enters myelin part of fiber
						imo = 1;
						nearestFiberIndex = i;
						//printf("Particle goes from outside to myelin\n");
					}
				} else if (curDistToCenter < inRadPropSq*radSq) {				// Particle tries to enter inner fiber from outside	
					if (u < 1-m_permeability) {					// Particle cannot enter fiber
						//printf("Particle outside, target in core, reflected from outer fiber, curDistToCenter: %g, tube index: %u\n", curDistToCenter, i);
						//printf("Fiber pos: x: %g, y: %g, r2: %g\n", fiberPos.x, fiberPos.y, radSq);
						reflected = true;
						//pos = reflectedPos(oPos,pos,fiberPos,radSq,imo,1,inRadPropSq);
						pos = oPos;
						//printf("New position: x: %g, y: %g, z: %g\n", pos.x, pos.y, pos.z);
					}
					else if ((1-m_permeability < u) & (u < 1-m_permeability*m_permeability)){	// Particle enters myelin but cannot enter inner fiber
						//printf("Particle outside, target in core, reflected from inner fiber, curDistToCenter: %g, tube index: %u\n", curDistToCenter, i);
						//printf("Fiber pos: x: %g, y: %g, r2: %g\n", fiberPos.x, fiberPos.y, radSq);
						reflected = true;
						//pos = reflectedPos(oPos,pos,fiberPos,radSq,1,1,inRadPropSq);
						pos = oPos;
						nearestFiberIndex = i;						
						//printf("New position: x: %g, y: %g, z: %g\n", pos.x, pos.y, pos.z);
						imo = 1;
					}
					else {								// Particle enters inner fiber
						imo = 0;
						nearestFiberIndex = i;
						//printf("Particle goes from outside to core\n");
					}
				} // end(if)
			}// end(for)

			// Temporary code: Checks if the particle bounces into another fiber and changes nearestFiberIndex accordingly. Need to clean this up!
			cubePos = calcCubePos(pos);
			cubeIndex = calcCubeHash(cubePos);
			float newDistToCenter;  		      

			if (imo == 2) {		
				for(uint j=0;j<m_hCubeCounter[cubeIndex];j++){
					float3 fiberPos;
					int i = (m_hCubes[cubeIndex* m_maxFibersPerCube+j]);
					fiberPos.x = m_fiberPos[i*4];
					fiberPos.y = m_fiberPos[i*4+1];
					fiberPos.z = m_fiberPos[i*4+2];
					radSq = m_fiberPos[i*4+3];

					newDistToCenter = distToCenterSq(pos,fiberPos);
					if (newDistToCenter < inRadPropSq*radSq) {
						//printf("Bounces into fiber %u core! fiberPos.x: %g, fiberPos.y: %g, radSq: %g \n", i, fiberPos.x, fiberPos.y, radSq);
						//cin.get();
						//imo = 0;
						pos = oPos;
						//nearestFiberIndex = i;
					} else if (newDistToCenter < radSq) {
						//printf("Bounces into fiber %u myelin! fiberPos.x: %g, fiberPos.y: %g, radSq: %g \n", i, fiberPos.x, fiberPos.y, radSq);
						//cin.get();
						//imo = 1;
						pos = oPos;
						//nearestFiberIndex = i;
					}
				}
			}	// end of temporary code
		}// end(else - particle outside)

	


	} // end if(permeability)

	


	// don't let the spin leave the volume

	if (pos.x >  1.0f - m_spinRadius) { pos.x =  1.0f - m_spinRadius; }
	else if (pos.x < -1.0f + m_spinRadius) { pos.x = -1.0f + m_spinRadius; }
	if (pos.y >  1.0f - m_spinRadius) { pos.y =  1.0f - m_spinRadius; }
	else if (pos.y < -1.0f + m_spinRadius) { pos.y = -1.0f + m_spinRadius; }
	if (pos.z >  1.0f - m_spinRadius) { pos.z =  1.0f - m_spinRadius; }
	else if (pos.z < -1.0f + m_spinRadius) { pos.z = -1.0f + m_spinRadius; }


	// store new position
	m_hPos[index*4] = pos.x;								// Change 1/28: Changed 4 to 5
	m_hPos[index*4+1] = pos.y;								// Change 1/28: Changed 4 to 5
	m_hPos[index*4+2] = pos.z;								// Change 1/28: Changed 4 to 5

	// Calculate change in field magnitude of each spin
	switch (imo){
		case 0:
			m_hPartParams[index*m_nParams] += -m_hPartParams[index*m_nParams]/m_T2fiber*deltaTime;
			break;
		case 1:
			m_hPartParams[index*m_nParams] += -m_hPartParams[index*m_nParams]/m_T2myelin*deltaTime;
			break;
		case 2:
			m_hPartParams[index*m_nParams] += -m_hPartParams[index*m_nParams]/m_T2outside*deltaTime;
			break;
	}


	// Calculate change in field phase of each spin
	m_hPartParams[index*m_nParams+1] += (magneticGradient.x * pos.x + magneticGradient.y * pos.y + magneticGradient.z * pos.z) * phaseConstant;

	m_hPartParams[index*m_nParams+2] = nearestFiberIndex;
	m_hPartParams[index*m_nParams+3] = imo;

	// Update random seeds
	m_hSeed[index*4] = seed[0];
	m_hSeed[index*4+1] = seed[1];
	m_hSeed[index*4+3] = nearestFiberIndex;

	//printf("*\nFinished with cpuIntegrate in spinKernelCpu.cpp\n*\n");	
}
