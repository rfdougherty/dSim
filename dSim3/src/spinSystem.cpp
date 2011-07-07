////////////////////////////////////////////////////////////////////////////////////////////////////
// File name: spinSystem.cpp
// Description: Defines the class spinSystem, on which the diffusion simulation is performed.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string.h>
#include <cstdlib>
#include <limits.h>
#include <GL/glew.h>
#include <time.h>
#include <algorithm>

#include "spinSystem.h"
#include "spinSystem.cuh"
#include "RTree.h"
//#include "dSimDataTypes.h"

#define PI 3.14159265358979f
#define TWOPI 6.28318530717959f


/////////////////////////////////////////////////////////////////////////////////////////////////////
// Function name:	frand
// Description:		Returns a random floating value in the range [0,1].
/////////////////////////////////////////////////////////////////////////////////////////////////////
inline float frand(){
	return rand() / (float) RAND_MAX;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////
// Function name:	SpinSystem
// Description:		SpinSystem constructor. The prefix m_ is used for variables global to the class
//			(i.e. defined in the header file). m_h and m_d are used to distinguish between
//			pointers to host and device memory, respectively.
//////////////////////////////////////////////////////////////////////////////////////////////////////
SpinSystem::SpinSystem(int numSpins, bool useGpu, float spaceScale, float gyroMagneticRatio, bool useDisplay, uint triSearchMethod, uint reflectionType, float extraAdc, 
			float myelinAdc, float intraAdc, float permeability, float extraT2, float myelinT2, float intraT2, float deltaTime, float startBoxSize){
	m_numSpins = numSpins;					// The number of spins in the simulation.
	m_nPosValues = 3;					// The number of parameters per spin in the position vector (x,y,z and color)
	m_nSeedValues = 2;					// The number of parameters per spin in the seed vector
	//m_nSpinValues = 4;					// The number of parameters per spin in the spin information vector (compartment, signal magnitude, signal phase and fiber index).
	m_numCompartments = 4;					// Number of compartments in the simulation (should ideally be read from the fiber file).
	m_nMembraneTypes = 3;
	m_deltaTime = deltaTime;
	m_hT2Values = new float[m_numCompartments];		// Array for storing the T2 value of each compartment
	m_hT2Values[0] = extraT2; 
	m_hT2Values[1] = intraT2, 
	m_hT2Values[2] = myelinT2;
	m_hT2Values[3] = intraT2;
	m_hStdDevs = new float[m_numCompartments];		// Array for storing the standard deviation of each compartment
	m_hStdDevs[0] = sqrt(2*extraAdc/(spaceScale*spaceScale)*deltaTime); 	// Adc is in um^2/msec. deltaTime is in msec, so only the space needs to be scaled
	m_hStdDevs[1] = sqrt(2*intraAdc/(spaceScale*spaceScale)*deltaTime); 
	m_hStdDevs[2] = sqrt(2*myelinAdc/(spaceScale*spaceScale)*deltaTime); 
	m_hStdDevs[3] = sqrt(2*intraAdc/(spaceScale*spaceScale)*deltaTime); 
	m_nAxonFibers = 0;
	m_nMyelinFibers = 0;
	m_nGliaFibers = 0;
	m_useGpu = useGpu;					// 1 if we perform spin computations on GPU, 0 otherwise
	m_startBoxSize = startBoxSize;				// The spins start in a box extending this fraction of the simulation volume in all directions.
	m_spinRadius = 0.005;					// Spin radius - this is only used for rendering.
	m_permeability = permeability;				// Probability of spin permeating membrane during collision
	m_spaceScale = spaceScale;				// The volume goes from -m_spaceScale to +m_spaceScale in all directions
	m_gradient = make_float3(0,0,0);			// The current gradient - defaulted to zero in the beginning.
	m_reflectionType = reflectionType;			// 1 if we have "realistic" collisions, 0 if we have simplified collisions
	m_triSearchMethod = triSearchMethod;			// 1 if we use R-Tree for collision detection (preferred), 0 if we use rectangular grid
	//m_numCubes = 50;					// The number of "cubes" in each direction in the rectangular grid - should be calculated from triangle sizes.
	//m_totalNumCubes = m_numCubes*m_numCubes*m_numCubes;	// Total number of cubes in the rectangular grid.
	//m_maxTrianglesPerCube = 25;				// Maximum number of triangles assigned to each cube - should be determined dynamically in populateCubes().
	//m_cubeLength = 2.0f / m_numCubes;			// The length of each cube, assuming a normalized volume (ranging from -1 to 1)
	m_xmax = m_ymax = m_zmax = 1.0f;			// Will denote the absolute value of the maximum x,y,z coordinates of the fibers (after scaling)
	m_gyroMagneticRatio = gyroMagneticRatio;
	m_useDisplay = useDisplay;				// 1 if simulation is displayed graphically on computer screen, 0 otherwise.
	m_currentPosRead = 0;
	m_currentSeedRead = 0;
	m_currentPosWrite = 1;
	m_currentSeedWrite = 1;
	m_currentSpinRead = 0;
	m_currentSpinWrite = 1;
	m_dPos[0] = m_dPos[1] = 0;
	m_dSeed[0] = m_dSeed[1] = 0;
	m_dSpins[0] = m_dSpins[1] = 0;		

	srand(time(0));
}


///////////////////////////////////////////////////////////////////
// Function name:	~SpinSystem
// Description:		SpinSystem destructor.
///////////////////////////////////////////////////////////////////
SpinSystem::~SpinSystem(){
	_finalize();
	m_numSpins = 0;

}


//////////////////////////////////////////////////////////////////
// Function name:	_finalize
// Description:		Frees up memory used by spinSystem class,
//			called by class destructor.
//////////////////////////////////////////////////////////////////
void SpinSystem::_finalize(){
	printf("Calling spin system destructor\n");
	//unbindTrgls();
	unbindVertices();
	unbindTriangleHelpers();
	unbindTriInfo();
	

	delete [] m_hPos;
	delete [] m_hSeed;
	delete [] m_hSpins;
	delete [] m_vertices;
	delete [] m_trgls;
	delete [] m_nTrianglesInMembraneType;
	delete [] m_triangleHelpers;
	delete [] m_triCounter;
	delete [] m_fibers;
	delete [] m_hTriInfo;
	delete [] m_hT2Values;
	delete [] m_hStdDevs;

	if (m_triSearchMethod == 1){
		unbindRTreeArray();
		unbindTreeIndexArray();
		delete [] m_treeGroup;
	}

	//freeArray(&m_dSeed[0]);
	//freeArray(&m_dSeed[1]);

	//freeArray(&m_dSpins[0]);
	//freeArray(&m_dSpins[1]);

	freeArray(m_dSeed[0]);
	freeArray(m_dSeed[1]);

	freeArray(m_dSpins[0]);
	freeArray(m_dSpins[1]);

	if (m_useDisplay){
		unregisterGLBufferObject(m_posVbo[0]);
		unregisterGLBufferObject(m_posVbo[1]);
		glDeleteBuffers(2, m_posVbo);
		glDeleteBuffers(1, &m_colorVBO);
	} else{
		freeArray(m_dPos[0]);
		freeArray(m_dPos[1]);
	}
	printf("Done calling spin system destructor\n");
}


////////////////////////////////////////////////////////////////////
// Function name:	createVBO
// Description:		Creates Vertex Buffer Object, which will be
//			used for OpenGL processing of the spin positions,
//			if the simulation is displayed on the screen.
////////////////////////////////////////////////////////////////////
GLuint SpinSystem::createVBO(uint size){
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	registerGLBufferObject(vbo);
	return vbo;
}


///////////////////////////////////////////////////////////////////////
// Function name:	build
// Description:		Allocates memory for arrays in host and device
///////////////////////////////////////////////////////////////////////
bool SpinSystem::build(){
	
	///////////////////////////////////////////////////////////////
	// Allocate host storage for position, seed and spin info arrays.
	// The information contained in the spins array is:
	//	m_hSpins[n].compartmentType:	Compartment number. For 3 compartments, 
	//			determines whether spin is outside fibers, inside axons, in myelin or in glia (0,1,2,3);
	//	m_hSpins[n].signalMagnitude:	Signal magnitude; 
	//	m_hSpins[n].signalPhase:	Signal phase;
	//	m_hSpins[n].insideFiber:	Fiber which the spin is inside of, UINT_MAX if not inside any fiber
	///////////////////////////////////////////////////////////////
	m_hPos = new float[m_numSpins*m_nPosValues];
	memset(m_hPos, 0, m_numSpins*m_nPosValues*sizeof(float));
	m_hSeed = new uint[m_numSpins*m_nSeedValues];
	memset(m_hSeed, 0, m_numSpins*m_nSeedValues*sizeof(uint));
	//m_hSpins = new float[m_numSpins*m_nSpinValues];
	//memset(m_hSpins, 0, m_numSpins*m_nSpinValues*sizeof(float));
	m_hSpins = new spinData[m_numSpins];
	memset(m_hSpins, 0, m_numSpins*sizeof(spinData));

	////////////////////////////////////////////////////////////////
	// Allocate memory for m_hCubeCounter and m_hTrianglesInCubes.
	// 	Sizes:	m_hCubeCounter [m_nMembraneTypes][m_nCubes];
	//		m_hTrianglesInCubes [m_nMembraneTypes][m_nCubes][m_maxTrianglesPerCube];
	//
	// 		m_hCubeCounter[i]j] = Number of triangles in cube j, belonging to membrane type no. i
	//		m_hTrianglesInCubes[i][j][k] = Triangle number k in cube j, belonging to membrane type no. i
	/////////////////////////////////////////////////////////////////
	/*m_hCubeCounter = new uint[m_nMembraneTypes*m_totalNumCubes];
	memset(m_hCubeCounter, 0, m_nMembraneTypes*m_totalNumCubes*sizeof(uint));
	m_hTrianglesInCubes = new uint[m_nMembraneTypes*m_totalNumCubes*m_maxTrianglesPerCube];
	memset(m_hTrianglesInCubes, 0, m_nMembraneTypes*m_totalNumCubes*m_maxTrianglesPerCube*sizeof(uint));*/
//	m_hCubeCounter = new uint[m_totalNumCubes];
	

	/////////////////////////////////////////////////////////////////
	// Allocate GPU data
	/////////////////////////////////////////////////////////////////
	if (m_useDisplay){
		m_posVbo[0] = createVBO(sizeof(float) * m_nPosValues * m_numSpins);
		m_posVbo[1] = createVBO(sizeof(float) * m_nPosValues * m_numSpins);

		m_colorVBO = createVBO(m_numSpins*m_nPosValues*sizeof(float));

		// Fill color buffer
		glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
		float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		float *ptr = data;
		for (uint i=0; i<m_numSpins; i++){
			*ptr++ = 0.5f;
			*ptr++ = 0.5f;
			*ptr++ = 0.5f;
			//*ptr++ = 0.5f;	// For spin
		}
		glUnmapBufferARB(GL_ARRAY_BUFFER);
	} else {
		allocateArray((void**)&m_dPos[0], sizeof(float) * m_nPosValues * m_numSpins);
		allocateArray((void**)&m_dPos[1], sizeof(float) * m_nPosValues * m_numSpins);
	}

	//allocateArray((void**)&m_dSpins[0], sizeof(float) * m_nSpinValues * m_numSpins);
	//allocateArray((void**)&m_dSpins[1], sizeof(float) * m_nSpinValues * m_numSpins);
	allocateArray((void**)&m_dSpins[0], m_numSpins*sizeof(spinData));
	allocateArray((void**)&m_dSpins[1], m_numSpins*sizeof(spinData));

	allocateArray((void**)&m_dSeed[0], sizeof(uint) * m_nSeedValues * m_numSpins);
	allocateArray((void**)&m_dSeed[1], sizeof(uint) * m_nSeedValues * m_numSpins);

	allocateArray((void**)&m_dT2Values, m_numCompartments*sizeof(float));
	copyArrayToDevice(m_dT2Values, m_hT2Values, 0, m_numCompartments*sizeof(float));

	allocateArray((void**)&m_dStdDevs, m_numCompartments*sizeof(float));
	copyArrayToDevice(m_dStdDevs, m_hStdDevs, 0, m_numCompartments*sizeof(float));


	// Set constants in device memory

	char gpuName_reflectionType [] = "k_reflectionType";
	copyConstantToDevice(gpuName_reflectionType, &m_reflectionType, 0, sizeof(uint));

	char gpuName_triSearchMethod [] = "k_triSearchMethod";
	copyConstantToDevice(gpuName_triSearchMethod, &m_triSearchMethod, 0, sizeof(uint));

	char gpuName_nFibers [] = "k_nFibers";
	copyConstantToDevice(gpuName_nFibers, &m_nFibers, 0, sizeof(uint));

	char gpuName_nCompartments [] = "k_nCompartments";
	copyConstantToDevice(gpuName_nCompartments, &m_numCompartments, 0, sizeof(uint));

	char gpuName_permeability [] = "k_permeability";
	copyConstantToDevice(gpuName_permeability, &m_permeability, 0, sizeof(float));

	char gpuName_deltaTime [] = "k_deltaTime";
	copyConstantToDevice(gpuName_deltaTime, &m_deltaTime, 0, sizeof(float));

	char gpuName_T2Values [] = "k_T2Values";
	copyConstantToDevice(gpuName_T2Values, &m_dT2Values, 0, sizeof(float));

	char gpuName_StdDevs [] = "k_stdDevs";
	copyConstantToDevice(gpuName_StdDevs, &m_dStdDevs, 0, sizeof(float));

	
	return(true);
}


////////////////////////////////////////////////////////////////////////
// Function name:	initFibers
// Inputs:		fiberFile: Name of file containing fiber triangle mesh
// Description:		Uses the mesh description in fiberFile to create 
//			arrays containing vertices, triangles and fibers
////////////////////////////////////////////////////////////////////////
bool SpinSystem::initFibers( char * fiberFile){
	

	FILE *fiberFilePtr = fopen(fiberFile,"r");
	if (fiberFilePtr == NULL){
		printf("Invalid name of fiber file.\n");
		exit(1);
	}
	float x, y, z, v1, v2, v3, axonTri, myelinTri, gliaTri, axonNr, myelinNr, gliaNr;
	int nScanned, posVar;
	bool doneReadingFiberFile = false;
	bool doneReadingFiberHeader = false;
	uint i;

	/////////////////////////////////////////////////////////////////
	// Data type can be V, T, A or M for vertex, triangle, axon or myelin
	// First we read the header of the input file and set the dataType to
	// a dummy value (X). Inputstring contains a short description of the
	// data being read in the header.
	/////////////////////////////////////////////////////////////////
	char dataType = 'X';
	/*char inputString[30];							// Note: Should try to not constrain the length of the input string
	// Read number of vertices
	nScanned = fscanf(fiberFilePtr, "%s", &inputString);
	nScanned = fscanf(fiberFilePtr, "%u", &m_nVertices);
	// Read number of triangles
	nScanned = fscanf(fiberFilePtr, "%s", &inputString);
	nScanned = fscanf(fiberFilePtr, "%u", &m_nTriangles);
	// Read number of membrane types
	uint m_nMembraneTypes_temp;						// Will delete
	nScanned = fscanf(fiberFilePtr, "%s", &inputString);
	nScanned = fscanf(fiberFilePtr, "%u", &m_nMembraneTypes_temp);
	// Read number of fibers
	nScanned = fscanf(fiberFilePtr, "%s", &inputString);
	nScanned = fscanf(fiberFilePtr, "%u", &m_nFibers);
	// Read the maximum number of triangles belonging to a single membrane
	nScanned = fscanf(fiberFilePtr, "%s", &inputString);
	nScanned = fscanf(fiberFilePtr, "%u", &m_maxTrianglesOnSurface);*/

	//////////////////////////////////////////////////////////////////////
	// Use the information from the header to allocate memory for the following
	// arrays:
	//		m_vertices:	m_vertices[i][0,1,2]: x,y,z coordinates 
	//				of vertex no. i
	//		m_trgls:	m_trgls[i][0,1,2]: Indexes of the vertices 
	//				(in m_vertices) that define triangle i
	//		m_triangleHelpers: Precomputed vector values for each triangle, to avoid
	//				   having to compute them over and over. See explanation
	//				   during creation of m_triangleValues
	//		m_triCounter:	m_triCounter[i][j]: Number of triangles 
	//				in membrane type i belonging to fiber j
	//		m_fibers:	m_fibers[i][j][k]: Index of triangle k in 
	//				membrane type i belonging to fiber j
	//		m_hTriInfo: m_hTriInfo[i][0]: The fiber which triangle i belongs to
	//				     m_hTriInfo[i][0]: The membrane type which triangle i belongs to
	///////////////////////////////////////////////////////////////////////
	
	//m_vertices = new float[m_nVertices*3];
	//memset(m_vertices, 0, m_nVertices*3*sizeof(float));
	//m_trgls = new uint[m_nTriangles*3];
	//memset(m_trgls, 0, m_nTriangles*3*sizeof(uint));
	m_nTrianglesInMembraneType = new uint[m_nMembraneTypes];
	memset(m_nTrianglesInMembraneType, 0, m_nMembraneTypes*sizeof(uint));
	//m_triangleHelpers = new float[m_nTriangles*12];
	//memset(m_triangleHelpers, 0, m_nTriangles*12*sizeof(float));
	//m_triCounter = new uint[m_nMembraneTypes*m_nFibers];
	//memset(m_triCounter, 0, m_nMembraneTypes*m_nFibers*sizeof(uint));
	//m_fibers = new uint[m_nMembraneTypes*m_nFibers*m_maxTrianglesOnSurface];
	//memset(m_fibers, 0, m_nMembraneTypes*m_nFibers*m_maxTrianglesOnSurface*sizeof(uint));
	//m_hTriInfo = new uint[m_nTriangles*3];
	//memset(m_hTriInfo, 0, m_nTriangles*3*sizeof(uint));

	//m_nMembraneTypes_temp = 0;
	//m_nMaxTrianglesPerMembrane_temp = 0; 
	//m_nFibers_temp = 0;
	m_nVertices = 0;
	m_nTriangles = 0;
	m_maxTrianglesOnSurface = 0;
	m_nFibers = 0;

	printf("m_nVertices: %u\n", m_nVertices);
	printf("m_nTriangles: %u\n", m_nTriangles);
	printf("m_maxTrianglesOnSurface: %u\n", m_maxTrianglesOnSurface);
	printf("m_nFibers: %u\n", m_nFibers);
	printf("m_nAxonFibers: %u\n", m_nAxonFibers);
	printf("m_nMyelinFibers: %u\n", m_nMyelinFibers);
	printf("m_nGliaFibers: %u\n", m_nGliaFibers);
	///////////////////////////////////////////////////////////////
	// Start reading the mesh
	///////////////////////////////////////////////////////////////

	//printf("Position indicator: %i\n", ftell(fiberFilePtr));
	while (!doneReadingFiberFile){

		if (doneReadingFiberHeader){

			m_nFibers = m_nAxonFibers+m_nGliaFibers;

			m_vertices = new float[m_nVertices*3];
			memset(m_vertices, 0, m_nVertices*3*sizeof(float));
			m_trgls = new uint[m_nTriangles*3];
			memset(m_trgls, 0, m_nTriangles*3*sizeof(uint));
			m_triangleHelpers = new float[m_nTriangles*12];
			memset(m_triangleHelpers, 0, m_nTriangles*12*sizeof(float));
			m_triCounter = new uint[m_nMembraneTypes*m_nFibers];
			memset(m_triCounter, 0, m_nMembraneTypes*m_nFibers*sizeof(uint));
			m_fibers = new uint[m_nMembraneTypes*m_nFibers*m_maxTrianglesOnSurface];
			memset(m_fibers, 0, m_nMembraneTypes*m_nFibers*m_maxTrianglesOnSurface*sizeof(uint));
			m_hTriInfo = new uint[m_nTriangles*3];
			memset(m_hTriInfo, 0, m_nTriangles*3*sizeof(uint));
		}

		// Start reading through the fiber file
		while (!feof(fiberFilePtr)){
			//printf("Start: Position indicator: %i\n", ftell(fiberFilePtr));
			nScanned = fscanf(fiberFilePtr, "%c", &dataType);
			//printf("Datatype: %c\n", dataType);
			//printf("(dataType == 'V': %i\n", dataType=='V');
			if (dataType == 'V'){
				//printf("V: Position indicator: %i\n", ftell(fiberFilePtr));
				// Data type is V, we are reading the vertices
				i = 0;
				nScanned = 3;
				while (nScanned == 3){
					// While we scan lines consisting of three floats, we write them into m_vertices
					nScanned = fscanf(fiberFilePtr, "%g %g %g", &x, &y, &z);
					if (nScanned == 3){
						//printf("Vertex line: Position indicator: %i\n", ftell(fiberFilePtr));
						if (!doneReadingFiberHeader){
							//m_nVertices++;
							m_nVertices++;
						} else{
							m_vertices[i*3+0] = x/m_spaceScale;
							m_vertices[i*3+1] = y/m_spaceScale;
							m_vertices[i*3+2] = z/m_spaceScale;
							m_xmax = fmaxf(fabs(x)/m_spaceScale,m_xmax);
							m_ymax = fmaxf(fabs(y)/m_spaceScale,m_ymax);
							m_zmax = fmaxf(fabs(z)/m_spaceScale,m_zmax);
							i++;
						}
					}
				}
				//m_nVertices_temp = i;
			} else if (dataType == 'T'){
				//printf("T: Position indicator: %i\n", ftell(fiberFilePtr));
				// Data type is T, we are reading the triangles
				i = 0;
				nScanned = 3;
				while (nScanned == 3){
					// While we scan lines consisting of three floats, we write them into m_trgls
					nScanned = fscanf(fiberFilePtr, "%g %g %g", &v1, &v2, &v3);
					if (nScanned == 3){
						if (!doneReadingFiberHeader){
							//m_nTriangles++;
							m_nTriangles++;
						} else{
							m_trgls[i*3+0] = (uint) v1;
							m_trgls[i*3+1] = (uint) v2;
							m_trgls[i*3+2] = (uint) v3;
							m_hTriInfo[i*3+2] = (uint) v1;
							i++;
						}
					}
					//m_nTriangles_temp = i;
				}
			} else if (dataType == 'A'){
				// Data type is A, we are reading an axon/myelin membrane
				if (!doneReadingFiberHeader){
					m_nAxonFibers++;
				}
				nScanned = fscanf(fiberFilePtr, "%g", &axonNr);
				// The line should look like AN, where N is the axon number (fiber number)
				nScanned = 1;
				i = 0;
				while (nScanned == 1){
					// While we scan lines consisting of single floats, we are still in the current membrane
					nScanned = fscanf(fiberFilePtr, "%g", &axonTri);
					if (nScanned == 1){
						if (!doneReadingFiberHeader){
							i++;
						} else{
							m_fibers[(0*m_nFibers+(uint)axonNr)*m_maxTrianglesOnSurface+i] = (uint) axonTri;
							//printf("m_fibers[%u]: %u\n", (0*m_nFibers+(uint)axonNr)*m_maxTrianglesOnSurface+i, m_fibers[(0*m_nFibers+(uint)axonNr)*m_maxTrianglesOnSurface+i]);
							m_triCounter[0*m_nFibers+(uint)axonNr] += 1;
							m_hTriInfo[(uint)axonTri * 3] = (uint) axonNr;
							m_hTriInfo[(uint)axonTri * 3 + 1] = 0;
							i++;
						}
					}
				}
				//if (m_nAxonFibers > m_nFibers_temp){
				//	m_nFibers_temp = m_nAxonFibers;
				//}
				if (i > m_maxTrianglesOnSurface){
					m_maxTrianglesOnSurface = i;
				}
			} else if (dataType == 'M'){
				// Data type is M, we are reading a myelin/outside membrane
				if (!doneReadingFiberHeader){
					m_nMyelinFibers++;
				}
				nScanned = fscanf(fiberFilePtr, "%g", &myelinNr);
				// The line should look like MN, where N is the myelin number (fiber number)
				nScanned = 1;
				i = 0;
				while (nScanned == 1){
					// While we scan lines consisting of single floats, we are still in the current membrane
					nScanned = fscanf(fiberFilePtr, "%g", &myelinTri);
					if (nScanned == 1){
						if (!doneReadingFiberHeader){
							i++;
						} else{
							//printf("m_maxTrianglesOnSurface: %u\n", m_maxTrianglesOnSurface);
							//printf("m_nFibers: %u\n", m_nFibers);
							//printf("myelinNr: %u\n", (uint)myelinNr);
							//printf("i: %u\n", i);
							m_fibers[(1*m_nFibers+(uint)myelinNr)*m_maxTrianglesOnSurface+i] = (uint) myelinTri;
							//printf("m_fibers[%u]: %u\n", (1*m_nFibers+(uint)axonNr)*m_maxTrianglesOnSurface+i, m_fibers[(1*m_nFibers+(uint)axonNr)*m_maxTrianglesOnSurface+i]);
							m_triCounter[1*m_nFibers+(uint)myelinNr] += 1;
							m_hTriInfo[(uint)myelinTri * 3] = (uint) myelinNr;
							m_hTriInfo[(uint)myelinTri * 3 + 1] = 1;
							i++;
						}
					}
				}
				//if (m_nMyelinFibers > m_nFibers_temp){
				//	m_nFibers_temp = m_nMyelinFibers;
				//}
				if (i > m_maxTrianglesOnSurface){
					m_maxTrianglesOnSurface = i;
				}
			} else if (dataType == 'G'){
				// Data type is G, we are reading a glia/outside membrane
				if (!doneReadingFiberHeader){
					m_nGliaFibers++;
				}
				nScanned = fscanf(fiberFilePtr, "%g", &gliaNr);
				// The line should look like MN, where N is the glia number (fiber number)
				nScanned = 1;
				i = 0;
				while (nScanned == 1){
					// While we scan lines consisting of single floats, we are still in the current membrane
					nScanned = fscanf(fiberFilePtr, "%g", &gliaTri);
					if (nScanned == 1){
						if (!doneReadingFiberHeader){
							i++;
						} else{
							m_fibers[(2*m_nFibers+(uint)gliaNr)*m_maxTrianglesOnSurface+i] = (uint) gliaTri;
							m_triCounter[2*m_nFibers+(uint)gliaNr] += 1;
							m_hTriInfo[(uint)gliaTri * 3] = (uint) gliaNr;
							m_hTriInfo[(uint)gliaTri * 3 + 1] = 2;
							i++;
						}
					}
				}
				//if (m_nGliaFibers > m_nFibers_temp){
				//	m_nFibers_temp = m_nGliaFibers;
				//}
				if (i > m_maxTrianglesOnSurface){
					m_maxTrianglesOnSurface = i;
				}
			}
		}

		if (!doneReadingFiberHeader){
			doneReadingFiberHeader = true;
			rewind(fiberFilePtr);
		} else {
			doneReadingFiberFile = true;
		}
	}
	fclose(fiberFilePtr);

	printf("m_nVertices: %u\n", m_nVertices);
	printf("m_nTriangles: %u\n", m_nTriangles);
	printf("m_maxTrianglesOnSurface: %u\n", m_maxTrianglesOnSurface);
	printf("m_nFibers: %u\n", m_nFibers);
	printf("m_nAxonFibers: %u\n", m_nAxonFibers);
	printf("m_nMyelinFibers: %u\n", m_nMyelinFibers);
	printf("m_nGliaFibers: %u\n", m_nGliaFibers);
	printf("m_fibers[%u]: %u\n", 0, m_fibers[0]);

	/*for (uint k = 0; k<m_nVertices; k++){
		printf("m_vertices[%u,:]: [%g,%g,%g]\n", k, m_vertices[k*3+0]*m_spaceScale, m_vertices[k*3+1]*m_spaceScale, m_vertices[k*3+2]*m_spaceScale);
	}

	for (uint k = 0; k<m_nTriangles; k++){
		printf("m_trgls[%u,:]: [%u,%u,%u]\n", k, m_trgls[k*3+0], m_trgls[k*3+1], m_trgls[k*3+2]);
	}

	printf("m_fibers[0,0,:]:");
	for (uint k = 0; k<m_maxTrianglesOnSurface; k++){
		printf("%u,", m_fibers[(0*m_nFibers+0)*m_maxTrianglesOnSurface+k]);
	}
	printf("\n");

	printf("m_fibers[1,0,:]:");
	for (uint k = 0; k<m_maxTrianglesOnSurface; k++){
		printf("%u,", m_fibers[(1*m_nFibers+0)*m_maxTrianglesOnSurface+k]);
	}
	printf("\n");*/

	
	if (m_nAxonFibers != m_nMyelinFibers){
		printf("Error in fiber structure: Must have same number of membranes of type 0 (axons) and type 1 (myelin).\n");
		exit(1);
	}

	//std::cin.get();

	//if (m_nAxonFibers > 0){
	//	m_nMembraneTypes_temp++;
	//}
	//if (m_nMyelinFibers > 0){
	//	m_nMembraneTypes_temp++;
	//}
	//if (m_nGliaFibers > 0){
	//	m_nMembraneTypes_temp++;
	//}

	

	// Create m_triangleHelpers
	float x1, y1, z1, x2, y2, z2, x3, y3, z3, nx, ny, nz, length; 
	uint p1, p2, p3;
	for (int i=0; i<m_nTriangles; i++){
		p1 = m_trgls[i*3+0]; p2 = m_trgls[i*3+1]; p3 = m_trgls[i*3+2];			//p1,p2,p3: Indices of points that define triangle i
		x1 = m_vertices[p1*3+0]; y1 = m_vertices[p1*3+1]; z1 = m_vertices[p1*3+2];	//x1,y1,z1: Coordinates of p1
		x2 = m_vertices[p2*3+0]; y2 = m_vertices[p2*3+1]; z2 = m_vertices[p2*3+2];	//x2,y2,z2: Coordinates of p2
		x3 = m_vertices[p3*3+0]; y3 = m_vertices[p3*3+1]; z3 = m_vertices[p3*3+2];	//x3,y3,z3: Coordinates of p3

		
		////////////////////////////////////////////////////////////////////////
		// triangleHelpers[i][0,1,2]: x,y and z coordinates of normal to triangle
		////////////////////////////////////////////////////////////////////////
		nx = (y2-y1)*(z3-z1)-(z2-z1)*(y3-y1);
		ny = (z2-z1)*(x3-x1)-(x2-x1)*(z3-z1);
		nz = (x2-x1)*(y3-y1)-(y2-y1)*(x3-x1);
		length = sqrt(nx*nx+ny*ny+nz*nz);						// Make the normal of unit length
		m_triangleHelpers[i*12+0] = nx/length;
		m_triangleHelpers[i*12+1] = ny/length;
		m_triangleHelpers[i*12+2] = nz/length;
		
		/////////////////////////////////////////////////////////////////////////
		// m_triangleHelpers[3,4,5]: x,y,z coordinate of vector u = (p3-p1)
		// m_triangleHelpers[6,7,8]: x,y,z coordinate of vector v = (p2-p1)
		// m_triangleHelpers[9,10,11]: Values of dot products uv, uu and vv
		//    These will all come in handy in the collision detection.
		/////////////////////////////////////////////////////////////////////////
		m_triangleHelpers[i*12+3] = x3-x1;
		m_triangleHelpers[i*12+4] = y3-y1;
		m_triangleHelpers[i*12+5] = z3-z1;
		m_triangleHelpers[i*12+6] = x2-x1;
		m_triangleHelpers[i*12+7] = y2-y1;
		m_triangleHelpers[i*12+8] = z2-z1;			
		m_triangleHelpers[i*12+9] = (x3-x1)*(x2-x1) + (y3-y1)*(y2-y1) + (z3-z1)*(z2-z1);
		m_triangleHelpers[i*12+10] = (x3-x1)*(x3-x1) + (y3-y1)*(y3-y1) + (z3-z1)*(z3-z1);
		m_triangleHelpers[i*12+11] = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
	}

	// Copy m_trgls, m_vertices and m_triangleHelpers to texture memory in device
	//allocateArray((void**)&m_dTrgls, m_nTriangles*3*sizeof(uint));
	//copyArrayToDevice(m_dTrgls, m_trgls, 0, m_nTriangles*3*sizeof(uint));
	//bindTrgls(m_dTrgls, m_nTriangles*3);

	allocateArray((void**)&m_dVertices, m_nVertices*3*sizeof(float));
	copyArrayToDevice(m_dVertices, m_vertices, 0, m_nVertices*3*sizeof(float));
	bindVertices(m_dVertices, m_nVertices*3);

	allocateArray((void**)&m_dTriangleHelpers, m_nTriangles*12*sizeof(float));
	copyArrayToDevice(m_dTriangleHelpers, m_triangleHelpers, 0, m_nTriangles*12*sizeof(float));
	bindTriangleHelpers(m_dTriangleHelpers, m_nTriangles*12);

	allocateArray((void**)&m_dTriInfo, m_nTriangles*3*sizeof(uint));
	copyArrayToDevice(m_dTriInfo, m_hTriInfo, 0, m_nTriangles*3*sizeof(uint));
	bindTriInfo(m_dTriInfo, m_nTriangles*3);

	// Sum all triangles belonging to each membrane type
	for (uint membraneIndex = 0; membraneIndex < m_nMembraneTypes; membraneIndex++){
		for(uint fiberIndex=0; fiberIndex<m_nFibers; fiberIndex++){
			m_nTrianglesInMembraneType[membraneIndex] += m_triCounter[membraneIndex*m_nFibers+fiberIndex];
		}
	}
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function: calcCubePos()
// Return type: float3
// Parameters: float3 point,
// Description: Function calculates the cube cell to which the given position belongs in uniform cube
//
//		Converts a position coordinate (ranging from (-1,-1,-1) to (1,1,1) to a cube coordinate
//		(ranging from (0,0,0) to (m_numCubes-1, m_numCubes-1, m_numCubes-1)).
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//float3 SpinSystem::calcCubePos(float3 point)
//{
//    float3 cubePos = floor((point + 1.0f) / m_cubeLength);
//    return cubePos;
//}

uint3 SpinSystem::calcCubePos(float3 p)
{
	//m_cubeLength = 1.0f;
	//m_numCubes = 2;
	//p.x = -1.5;

	uint3 cubePos;
	//cubePos.x = floor((p.x + 1.0f) / m_cubeLength);
	//cubePos.y = floor((p.y + 1.0f) / m_cubeLength);
	//cubePos.z = floor((p.z + 1.0f) / m_cubeLength);

	//printf("cubePos.x: %u\n", cubePos.x);

	//cubePos.x = std::max(0, std::min(cubePos.x, m_numCubes-1));
	//cubePos.y = std::max(0, std::min(cubePos.y, m_numCubes-1));
	//cubePos.z = std::max(0, std::min(cubePos.z, m_numCubes-1));
	

	//cubePos.x = std::max((uint)0,std::min(cubePos.x, m_numCubes-1));
	cubePos.x = (uint) std::max(0.0f,std::min((float) floor((p.x + 1.0f) / m_cubeLength), (float) (m_numCubes-1)));
	cubePos.y = (uint) std::max(0.0f,std::min((float) floor((p.y + 1.0f) / m_cubeLength), (float) (m_numCubes-1)));
	cubePos.z = (uint) std::max(0.0f,std::min((float) floor((p.z + 1.0f) / m_cubeLength), (float) (m_numCubes-1)));

	//printf("In calcCubePos\n");
	//printf("m_numCubes:%u\n", m_numCubes);
	//printf("cubePos.x: %u\n", cubePos.x);

	////std::cin.get();

	return cubePos;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function: calcCubeHash()
// Return type: int
// Parameters: float3 cubePos
// Description: Function calculates address in cube (index in list of cells) from position (clamping to edges)
//
//		Given the cube coordinate (ranging from (0,0,0) to (m_numCubes-1, m_numCubes-1, m_numCubes-1), 
//		calculates the cube index as shown in the figure (for m_numCubes = 2).
//		 __________________
//		|\	  \        \
//		| \   2	   \   3    \	
//		|  \________\________\
//		\  |\	     \	      \
//		|\ | \ _______\________\
//		| \|  |        |        |
//		|  \  |	       |        |
//		\0 |\ |   6    |   7    |
//		 \ | \|________|________|	y
//		  \|  |	       |	|	|
//		   \  |	       |	|	|____x	 
//		    \ |	  4    |   5	|	 \ 
//		     \|________|________|	  \z
//
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//int SpinSystem::calcCubeHash(float3 cubePos)
//{
//	cubePos = fmaxf(0.0f, fminf(m_numCubes-1, cubePos));		// m_numCubes is the number of cubes in any one dimension
//
//	return  ( (int)(m_numCubes*m_numCubes*cubePos.z) + (int)(m_numCubes*cubePos.y) + (int)cubePos.x);   	// Can take m_numCubes^3 values, from 0 to m_numCubes^3-1
//}

uint SpinSystem::calcCubeHash(uint3 cubePos)
{
	return  m_numCubes*m_numCubes*cubePos.z + m_numCubes*cubePos.y + cubePos.x;   	// Can take m_numCubes^3 values, from 0 to m_numCubes^3-1
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function: calcCubeInterSect()
// Return type: uint
// Description: Finds the cubes intersected by a ray from startPoint to endPoint. Writes the indices of the
//		cubes traversed by the ray into cubeArray. Returns the numbers of cubes in cubeArray.
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*uint SpinSystem::calcCubeIntersect(float3 startPoint, float3 endPoint, uint cubeArray[])
{
	uint startCubex, startCubey, startCubez, endCubex, endCubey, endCubez, numCubes;

	//////////////////////////////////////////////////////////////////////////
	// StartCubeIndex and EndCubeIndex are the indices of the cubes in which
	// the ray starts and ends, respectively
	//////////////////////////////////////////////////////////////////////////
	uint startCubeIndex = calcCubeHash(calcCubePos(startPoint));
	uint endCubeIndex = calcCubeHash(calcCubePos(endPoint));

	if (startCubeIndex == endCubeIndex){
		// The ray starts and ends in the same cube, no other cubes need to be checked.
		cubeArray[0] = startCubeIndex;
		numCubes = 1;
	} else{
		// Find the "cube coordinates" of the cubes in terms of number of cube lengths in each principal direction
		startCubex = startCubeIndex - (startCubeIndex/m_numCubes)*m_numCubes;						// cubex = mod(cubeIndex,m_numCubes)
		startCubey = (startCubeIndex - (startCubeIndex/(m_numCubes*m_numCubes))*m_numCubes*m_numCubes)/m_numCubes;	// cubey = floor(mod(cubeIndex,m_numCubes^2)/m_numCubes)
		startCubez = startCubeIndex/(m_numCubes*m_numCubes);								// cubez = floor(cubeIndex/m_numCubes^2)
		endCubex = endCubeIndex - (endCubeIndex/m_numCubes)*m_numCubes;							// cubex = mod(cubeIndex,m_numCubes)
		endCubey = (endCubeIndex - (endCubeIndex/(m_numCubes*m_numCubes))*m_numCubes*m_numCubes)/m_numCubes;		// cubey = floor(mod(cubeIndex,m_numCubes^2)/m_numCubes)
		endCubez = endCubeIndex/(m_numCubes*m_numCubes);								// cubez = floor(cubeIndex/m_numCubes^2)


		if ( ((startCubex == endCubex)&(startCubey == endCubey)) | ((startCubex == endCubex)&(startCubez == endCubez)) | ((startCubey == endCubey)&(startCubez == endCubez)) ){
			// The two cubes have all but one coordinate in common, so the ray only intersects these two cubes
			cubeArray[0] = startCubeIndex;
			cubeArray[1] = endCubeIndex;
			numCubes = 2;
		} else if (startCubez == endCubez){	// The start/end cubes have only the same coordinate in the z-direction. We add the two cubes in the x- and y-directions
			cubeArray[0] = startCubeIndex;
			cubeArray[1] = endCubeIndex;
			cubeArray[2] = calcCubeHash(make_float3(startCubex,endCubey,endCubez));
			cubeArray[3] = calcCubeHash(make_float3(startCubex,endCubey,startCubez));
			numCubes = 4;
		} else if (startCubex == endCubex){	// The start/end cubes have only the same coordinate in the x-direction. We add the two cubes in the y- and z-directions
			cubeArray[0] = startCubeIndex;
			cubeArray[1] = endCubeIndex;
			cubeArray[2] = calcCubeHash(make_float3(endCubex,startCubey,endCubez));
			cubeArray[3] = calcCubeHash(make_float3(startCubex,endCubey,startCubez));
			numCubes = 4;
		} else if (startCubey == endCubey){	// The start/end cubes have only the same coordinate in the y-direction. We add the two cubes in the x- and z-directions
			cubeArray[0] = startCubeIndex;
			cubeArray[1] = endCubeIndex;
			cubeArray[2] = calcCubeHash(make_float3(startCubex,endCubey,endCubez));
			cubeArray[3] = calcCubeHash(make_float3(endCubex,startCubey,startCubez));
			numCubes = 4;
		} else {		// The start/end cubes have no coordinate in common - we add the six cubes contained within the 8-cube defined by the start/end cubes
			cubeArray[0] = startCubeIndex;
			cubeArray[1] = endCubeIndex;
			cubeArray[2] = calcCubeHash(make_float3(endCubex,endCubey,startCubez));
			cubeArray[3] = calcCubeHash(make_float3(endCubex,startCubey,endCubez));
			cubeArray[4] = calcCubeHash(make_float3(startCubex,endCubey,endCubez));
			cubeArray[5] = calcCubeHash(make_float3(endCubex,startCubey,startCubez));
			cubeArray[6] = calcCubeHash(make_float3(startCubex,endCubey,startCubez));
			cubeArray[7] = calcCubeHash(make_float3(startCubex,startCubey,endCubez));
			numCubes = 8;
		}
	}

	return numCubes;
}*/


void SpinSystem::constructAllRTrees(){
	
	uint *fibersInTree = new uint[m_nFibers];				// fibersInTree[f] = 1: The tree contains triangles belonging to fiber no. f
	uint *membranesInTree = new uint[m_nMembraneTypes];			// membranesInTree[m] = 1: The tree contains triangles belonging to membrane type m
	float *combinedTreeArray;
	uint *combinedTreeIndex;
	uint *treeArraySizes;
	uint numTrees, combinedTreeArraySize = 0;

	uint searchMethod = 0;


	printf("Constructing totalTree\n");
	// Create a tree containing all triangles - this is useful for the function findSpinsInFibers
	memset(fibersInTree,0,m_nFibers*sizeof(uint));
	memset(membranesInTree,0,m_nMembraneTypes*sizeof(uint));
	for (uint f=0; f<m_nFibers; f++){
		fibersInTree[f] = 1;
	}
	for (uint m=0; m<m_nMembraneTypes; m++){
		membranesInTree[m] = 1;
	}
	createRTree(totalTree,membranesInTree,fibersInTree);
	printf("Done constructing totalTree\n");




	// Start 'if searchMethod == RTree'
	if (searchMethod==1){
		numTrees = m_nFibers*(m_numCompartments-1)+1;			// We create a tree for each compartment and each fiber - some trees will be empty
		treeArraySizes = new uint[numTrees];				// treeArraySizes[t] = N: When converted to a float array, the tree t is of size N
		combinedTreeIndex = new uint[numTrees];				// combinedTreeIndex[t] = N: After we construct one big array out of all the tree arrays,
											//			then tree t will start at index N in the array.
		m_treeGroup = new RTree<uint, float, 3, float>[numTrees];		// m_treeGroup is an array of trees, each compartment will be assigned a tree.
		uint totalTreeArraySize;

		printf("Number of trees: %u\n", numTrees);

		// Create a tree containing the outermost membrane of all the fibers (to use for particles not inside any fiber)
		memset(membranesInTree,0,m_nMembraneTypes*sizeof(uint));
		membranesInTree[1] = 1;
		membranesInTree[2] = 1;
		memset(fibersInTree,0,m_nFibers*sizeof(uint));
		for (uint f=0; f<m_nFibers; f++){
			fibersInTree[f] = 1;
		}
		createRTree(m_treeGroup[0],membranesInTree,fibersInTree);


		// Create a tree for every compartment of every fiber (exclude the outermost compartment, representing the outside of all fibers)
		for (uint cmptIndex=1; cmptIndex<m_numCompartments; cmptIndex++){
			switch (cmptIndex){
				case 1:
					memset(membranesInTree,0,m_nMembraneTypes*sizeof(uint));
					membranesInTree[0] = 1;
					break;
				case 2:
					memset(membranesInTree,0,m_nMembraneTypes*sizeof(uint));
					membranesInTree[0] = 1;
					membranesInTree[1] = 1;
					break;
				case 3:
					memset(membranesInTree,0,m_nMembraneTypes*sizeof(uint));
					membranesInTree[2] = 1;
			}

			for (uint fiberIndex=0; fiberIndex<m_nFibers; fiberIndex++){
				memset(fibersInTree,0,m_nFibers*sizeof(uint));
				fibersInTree[fiberIndex] = 1;
				printf("compartment: %u, fiber: %u\n", cmptIndex, fiberIndex);
				printf("fiberIndex*(m_numCompartments-1)+cmptIndex: %u\n", fiberIndex*(m_numCompartments-1)+cmptIndex);
				createRTree(m_treeGroup[fiberIndex*(m_numCompartments-1)+cmptIndex],membranesInTree,fibersInTree);
				printf("Tree index: %u\n", fiberIndex*(m_numCompartments-1)+cmptIndex);
				printf("Tree height for fiber %u and compartment %u: %i\n", fiberIndex, cmptIndex, m_treeGroup[fiberIndex*(m_numCompartments-1)+cmptIndex].GetHeight());
				printf("All tree rectangles for fiber %u and compartment %u: %i\n", fiberIndex, cmptIndex, m_treeGroup[fiberIndex*(m_numCompartments-1)+cmptIndex].CountAllRects());
				printf("Leaf tree rectangles for fiber %u and compartment %u: %i\n", fiberIndex, cmptIndex, m_treeGroup[fiberIndex*(m_numCompartments-1)+cmptIndex].Count());
			}			
		}

		for (uint t=0; t<numTrees; t++){	
			m_treeGroup[t].CreateIndexArray();
			treeArraySizes[t] = m_treeGroup[t].GetTreeArraySize();
			printf("treeArraySizes[%u]: %u\n", t, treeArraySizes[t]);
			combinedTreeArraySize += treeArraySizes[t];
				
			if (t==0){
				combinedTreeIndex[t] = 0;
			} else{
				combinedTreeIndex[t] = combinedTreeIndex[t-1]+treeArraySizes[t-1];
			}
		}
	
		
		printf("(in SpinSystem::constructAllRTrees): combinedTreeIndex: [");
		for (uint t=0; t<numTrees-1; t++){
			printf("%u,", combinedTreeIndex[t]);
		}
		printf("%u]\n", combinedTreeIndex[numTrees-1]);


		if (m_useGpu){
			printf("Creating combined tree array\n");
			//uint combinedTreeArraySize = treeArraySizes[0]+treeArraySizes[1]+treeArraySizes[2];
			combinedTreeArray = new float[combinedTreeArraySize];
			memset(combinedTreeArray,0,combinedTreeArraySize*sizeof(float));
			for (uint t=0; t<numTrees; t++){
				m_treeGroup[t].RTree2Array(combinedTreeArray,combinedTreeIndex[t]);
			}
			printf("Done creating combined tree array\n");
		}

	// Done with 'if searchMethod == RTree'
	} else {		// Create dummy values for combinedTreeIndex and combinedTreeArray
		numTrees = 1;
		treeArraySizes = new uint[numTrees];
		combinedTreeIndex = new uint[numTrees];
		combinedTreeIndex[0] = 0;
		combinedTreeArraySize = 1;
		combinedTreeArray = new float[combinedTreeArraySize];
		combinedTreeArray[0] = 0;

	}

	if (m_useGpu){
		printf("Copying combined tree array to GPU\n");
		allocateArray((void**)&m_dRTreeArray, combinedTreeArraySize*sizeof(float));
		copyArrayToDevice(m_dRTreeArray, combinedTreeArray,0,combinedTreeArraySize*sizeof(float));
		bindRTreeArray(m_dRTreeArray,combinedTreeArraySize);
		printf("Done copying combined tree array to GPU\n");
	
		printf("Copying tree index array to GPU\n");
		allocateArray((void**)&m_dTreeIndexArray, numTrees*sizeof(uint));
		copyArrayToDevice(m_dTreeIndexArray,combinedTreeIndex,0,numTrees*sizeof(uint));
		bindTreeIndexArray(m_dTreeIndexArray,numTrees);
		printf("Done copying tree index array to GPU\n");
	}


	delete [] fibersInTree;
	delete [] membranesInTree;
	delete [] treeArraySizes;
	delete [] combinedTreeIndex;
	delete [] combinedTreeArray;

	printf("(in SpinSystem::constructAllRTrees): Exiting constructAllRTrees\n");
	//std::cin.get();
}


void SpinSystem::createRTree(RTree<uint, float, 3, float> &tree, uint* membranesInTree, uint *fibersInTree){
	
	float3 v1, v2, v3;
	uint v1Index, v2Index, v3Index;
	float minxyz[3], maxxyz[3];
	uint membraneIndex = 0;

	for (uint membraneIndex=0; membraneIndex<m_nMembraneTypes; membraneIndex++){
		if (membranesInTree[membraneIndex]){							// The type of membrane we are adding to the tree
			for (uint fiberIndex=0; fiberIndex<m_nFibers; fiberIndex++){
				if (fibersInTree[fiberIndex]){
					//printf("Membrane %u of fiber %u is in tree\n", membraneIndex, fiberIndex);
					uint numTriangles = m_triCounter[membraneIndex*m_nFibers + fiberIndex];		// Number of triangles on fiber fiberIndex belonging to membrane type membraneIndex	
					for (uint n=0; n<numTriangles; n++){
						uint triangleIndex = m_fibers[(membraneIndex*m_nFibers+fiberIndex)*m_maxTrianglesOnSurface + n];
	
						v1Index = m_trgls[triangleIndex*3+0]; v2Index = m_trgls[triangleIndex*3+1]; v3Index = m_trgls[triangleIndex*3+2]; 
						v1.x = m_vertices[v1Index*3+0]; v1.y = m_vertices[v1Index*3+1]; v1.z = m_vertices[v1Index*3+2];
						v2.x = m_vertices[v2Index*3+0]; v2.y = m_vertices[v2Index*3+1]; v2.z = m_vertices[v2Index*3+2];
						v3.x = m_vertices[v3Index*3+0]; v3.y = m_vertices[v3Index*3+1]; v3.z = m_vertices[v3Index*3+2];
			
						minxyz[0] = v1.x;	// Find minx, miny, minz
						if (v2.x < minxyz[0]){minxyz[0] = v2.x;}
						if (v3.x < minxyz[0]){minxyz[0] = v3.x;}
						minxyz[1] = v1.y;
						if (v2.y < minxyz[1]){minxyz[1] = v2.y;}
						if (v3.y < minxyz[1]){minxyz[1] = v3.y;}
						minxyz[2] = v1.z;
						if (v2.z < minxyz[2]){minxyz[2] = v2.z;}
						if (v3.z < minxyz[2]){minxyz[2] = v3.z;}
	
						maxxyz[0] = v1.x;	// Find maxx, maxy, maxz
						if (v2.x > maxxyz[0]){maxxyz[0] = v2.x;}
						if (v3.x > maxxyz[0]){maxxyz[0] = v3.x;}
						maxxyz[1] = v1.y;
						if (v2.y > maxxyz[1]){maxxyz[1] = v2.y;}
						if (v3.y > maxxyz[1]){maxxyz[1] = v3.y;}
						maxxyz[2] = v1.z;
						if (v2.z > maxxyz[2]){maxxyz[2] = v2.z;}
						if (v3.z > maxxyz[2]){maxxyz[2] = v3.z;}
			
						tree.Insert(minxyz, maxxyz, triangleIndex);
					}
				}
			}
		}
	}
}


/////////////////////////////////////////////////////////////////////////////////////////////
// Function name:	SearchTreeArray
// Description:		Search in the R-Tree array for all data rectangles that overlap 
//			the argument rectangle.
//			We are using an iterative preorder traversal with a 
//			stack - see http://en.wikipedia.org/wiki/Tree_traversal
//			This function is currently not in use but a version is implemented in
//			the spinKernel.cu file.
// Inputs:		rect is a rectangle tested against the tree for intersection with leaf rectangles.
//			rect[0,1,2,3,4,5] = rect.x_min,rect.y_min,rect.z_min,rect.x_max,rect.y_max,rect.z_max
/////////////////////////////////////////////////////////////////////////////////////////////
uint SpinSystem::SearchTreeArray(float* rect, float* RTreeArray, uint* interSectArray)
{
	uint foundCount = 0;				// The number of rectangles found to intersect the rectangle defined by rect
	uint stackSize = 100;				// Stack size of 100 is more than enough
	uint stack[stackSize];				// We will use the stack for keeping track of the nodes being checked in the tree
	int stackIndex = 0;
	
	stack[stackIndex] = 0;			// We push the location of the root node in the RTreeArray onto the stack
	stackIndex++;

	uint currentNodeIndex;

	while (stackIndex > 0){				// Stop when we've emptied the stack
		stackIndex--;					// Pop the top node off the stack
		currentNodeIndex = stack[stackIndex];

		for (int m=RTreeArray[currentNodeIndex+1]-1; m>=0; m--){
			uint currentBranchIndex = currentNodeIndex+2 + m*7;

			//See if the branch rectangle overlaps with the input rectangle
			if (!(  RTreeArray[currentBranchIndex+1] > rect[3] ||		// branchRect.x_min > rect.x_max
				RTreeArray[currentBranchIndex+2] > rect[4] ||		// branchRect.y_min > rect.y_max
				RTreeArray[currentBranchIndex+3] > rect[5] ||		// branchRect.z_min > rect.z_max
				rect[0] > RTreeArray[currentBranchIndex+4] ||		// rect.x_min > branchRect.x_max
				rect[1] > RTreeArray[currentBranchIndex+5] ||		// rect.y_min > branchRect.y_max
				rect[2] > RTreeArray[currentBranchIndex+6] ))		// rect.z_min > branchRect.z_max
			{
				if (RTreeArray[currentNodeIndex] > 0){		// We are at an internal node - push the node pointed to in the branch onto the stack
					stack[stackIndex] = RTreeArray[currentBranchIndex];
					stackIndex++;
				} else {
					interSectArray[foundCount] = RTreeArray[currentBranchIndex]; // We are at a leaf - store corresponding triangle
					foundCount++;
				}
			}
		}
	}
	return foundCount;
}




/////////////////////////////////////////////////////////////////////////////////////////////////////
// Function name: 	populateCubes.
// Description:		Produce the matrix m_hTrianglesInCubes. Line no. i in this matrix lists the 
//			triangles that are wholly or partly inside cube no. i.
// 			Also produces the array m_hCubeCounter, which holds the number of triangles in each cube.
////////////////////////////////////////////////////////////////////////////////////////////////////
void SpinSystem::populateCubes(){

	float3 p1, p2, p3, minPoint, maxPoint;
	uint3 minCubePoint, maxCubePoint;
	uint pointCube, v1, v2, v3;
	bool determinedMaxTriPerCube = false;
	bool populatedCubes = false;
	m_maxTrianglesPerCube = 0;

	float max_stdDev = m_hStdDevs[0];
	for (uint i=1; i<m_numCompartments; i++){
		if (m_hStdDevs[i] > max_stdDev) {
			max_stdDev = m_hStdDevs[i];
		}
	}

	m_numCubes = floor(2.0f/(7.0f*max_stdDev));
	printf("(In SpinSystem::populateCubes): m_numCubes: %u\n", m_numCubes);

	m_numCubes = 50;					// The number of "cubes" in each direction in the rectangular grid - should be calculated from triangle sizes.
	m_totalNumCubes = m_numCubes*m_numCubes*m_numCubes;	// Total number of cubes in the rectangular grid.
	m_cubeLength = 2.0f / m_numCubes;			// The length of each cube, assuming a normalized volume (ranging from -1 to 1)
	//m_maxTrianglesPerCube = 25;				// Maximum number of triangles assigned to each cube - should be determined dynamically in populateCubes().

	m_hCubeCounter = new uint[m_totalNumCubes];

	printf("(In SpinSystem::populateCubes): m_nMembraneTypes: %u\n", m_nMembraneTypes);
	printf("(In SpinSystem::populateCubes): m_nFibers: %u\n", m_nFibers);
	printf("(In SpinSystem::populateCubes): m_nTriangles: %u\n", m_nTriangles);
	printf("(In SpinSystem::populateCubes): m_numCubes: %u\n", m_numCubes);
	printf("(In SpinSystem::populateCubes): m_totalNumCubes: %u\n", m_totalNumCubes);

	
	while (!populatedCubes){

		memset(m_hCubeCounter,0,m_totalNumCubes*sizeof(uint));
		if (determinedMaxTriPerCube){
			m_hTrianglesInCubes = new uint[m_totalNumCubes*m_maxTrianglesPerCube];
		}

		for (uint nTri=0; nTri<m_nTriangles; nTri++){
			v1 = m_trgls[nTri*3+0];
			v2 = m_trgls[nTri*3+1];
			v3 = m_trgls[nTri*3+2];

			p1 = make_float3(m_vertices[v1*3+0],m_vertices[v1*3+1],m_vertices[v1*3+2]);
			p2 = make_float3(m_vertices[v2*3+0],m_vertices[v2*3+1],m_vertices[v2*3+2]);
			p3 = make_float3(m_vertices[v3*3+0],m_vertices[v3*3+1],m_vertices[v3*3+2]);

			// Find the defining points of the bounding box of the triangle
			minPoint.x = p1.x;	// Find minx, miny, minz of the triangle
			if (p2.x < minPoint.x){minPoint.x = p2.x;}
			if (p3.x < minPoint.x){minPoint.x = p3.x;}
			minPoint.y = p1.y;
			if (p2.y < minPoint.y){minPoint.y = p2.y;}
			if (p3.y < minPoint.y){minPoint.y = p3.y;}
			minPoint.z = p1.z;
			if (p2.z < minPoint.z){minPoint.z = p2.z;}
			if (p3.z < minPoint.z){minPoint.z = p3.z;}

			maxPoint.x = p1.x;	// Find maxx, maxy, maxz of the triangle
			if (p2.x > maxPoint.x){maxPoint.x = p2.x;}
			if (p3.x > maxPoint.x){maxPoint.x = p3.x;}
			maxPoint.y = p1.y;
			if (p2.y > maxPoint.y){maxPoint.y = p2.y;}
			if (p3.y > maxPoint.y){maxPoint.y = p3.y;}
			maxPoint.z = p1.z;
			if (p2.z > maxPoint.z){maxPoint.z = p2.z;}
			if (p3.z > maxPoint.z){maxPoint.z = p3.z;}

			// Convert to cube coordinates
			minCubePoint = calcCubePos(minPoint);
			maxCubePoint = calcCubePos(maxPoint);

			//minCubePoint = fmaxf(0.0f, fminf(m_numCubes-1, minCubePoint));		ATTN: Change
			//maxCubePoint = fmaxf(0.0f, fminf(m_numCubes-1, maxCubePoint));

			// Check every grid cube contained within the bounding box of the triangle
			for (uint xc=minCubePoint.x; xc<=maxCubePoint.x; xc++){
				for (uint yc=minCubePoint.y; yc<=maxCubePoint.y; yc++){
					for (uint zc=minCubePoint.z; zc<=maxCubePoint.z; zc++){
						uint3 cubeXYZ;
						cubeXYZ.x = xc;
						cubeXYZ.y = yc;
						cubeXYZ.z = zc;
						uint cubeIndex = calcCubeHash(cubeXYZ);
						bool interSection = false;
						// We are now looking at the grid cube with cube coordinates (xc,yc,zc)
						float3 cubeMin = make_float3(-1+xc*m_cubeLength,-1+yc*m_cubeLength,-1+zc*m_cubeLength);
						float3 cubeMax = make_float3(-1+(xc+1)*m_cubeLength,-1+(yc+1)*m_cubeLength,-1+(zc+1)*m_cubeLength);

						// Check if any of the end points of the triangle are contained within the cube
						if ( pointInCube(p1, cubeMin, cubeMax) ||
					     	pointInCube(p2, cubeMin, cubeMax) ||
					     	pointInCube(p3, cubeMin, cubeMax) ){
							interSection = true;
						// Check if any of the cube edges intersect the triangle
						// (1 detection for each side of the cube = 12 detections)
						} else if (rayTriangleIntersect(make_float3(cubeMin.x,cubeMin.y,cubeMin.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z),p1,p2,p3) ||
							rayTriangleIntersect(make_float3(cubeMin.x,cubeMin.y,cubeMin.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z),p1,p2,p3) ||
							rayTriangleIntersect(make_float3(cubeMin.x,cubeMin.y,cubeMin.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z),p1,p2,p3) ||
							rayTriangleIntersect(make_float3(cubeMin.x,cubeMax.y,cubeMax.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z),p1,p2,p3) ||
							rayTriangleIntersect(make_float3(cubeMin.x,cubeMax.y,cubeMax.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z),p1,p2,p3) ||
							rayTriangleIntersect(make_float3(cubeMin.x,cubeMax.y,cubeMax.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z),p1,p2,p3) ||
							rayTriangleIntersect(make_float3(cubeMax.x,cubeMax.y,cubeMin.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z),p1,p2,p3) ||
							rayTriangleIntersect(make_float3(cubeMax.x,cubeMax.y,cubeMin.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z),p1,p2,p3) ||
							rayTriangleIntersect(make_float3(cubeMax.x,cubeMax.y,cubeMin.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z),p1,p2,p3) ||
							rayTriangleIntersect(make_float3(cubeMax.x,cubeMin.y,cubeMax.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z),p1,p2,p3) ||
							rayTriangleIntersect(make_float3(cubeMax.x,cubeMin.y,cubeMax.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z),p1,p2,p3) ||
							rayTriangleIntersect(make_float3(cubeMax.x,cubeMin.y,cubeMax.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z),p1,p2,p3)){
								interSection = true;
						// Check if any of the triangle edges intersect any of the sides of the cube (we split each side into two triangles and check
						// for intersection (6x2 detections for each side of the triangle = 36 detections)
						// Start by checking triangle edge from p1 to p2
						} else if (rayTriangleIntersect(p1,p2,make_float3(cubeMin.x,cubeMin.y,cubeMin.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z)) ||
							rayTriangleIntersect(p1,p2,make_float3(cubeMin.x,cubeMin.y,cubeMin.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z)) ||
							rayTriangleIntersect(p1,p2,make_float3(cubeMin.x,cubeMin.y,cubeMin.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z)) ||
							rayTriangleIntersect(p1,p2,make_float3(cubeMin.x,cubeMax.y,cubeMax.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z)) ||
							rayTriangleIntersect(p1,p2,make_float3(cubeMin.x,cubeMax.y,cubeMax.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z)) ||
							rayTriangleIntersect(p1,p2,make_float3(cubeMin.x,cubeMax.y,cubeMax.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z)) ||
							rayTriangleIntersect(p1,p2,make_float3(cubeMax.x,cubeMax.y,cubeMin.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z)) ||
							rayTriangleIntersect(p1,p2,make_float3(cubeMax.x,cubeMax.y,cubeMin.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z)) ||
							rayTriangleIntersect(p1,p2,make_float3(cubeMax.x,cubeMax.y,cubeMin.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z)) ||
							rayTriangleIntersect(p1,p2,make_float3(cubeMax.x,cubeMin.y,cubeMax.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z)) ||
							rayTriangleIntersect(p1,p2,make_float3(cubeMax.x,cubeMin.y,cubeMax.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z)) ||
							rayTriangleIntersect(p1,p2,make_float3(cubeMax.x,cubeMin.y,cubeMax.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z)) ||
							// Repeat for triangle edge from p1 to p3
							rayTriangleIntersect(p1,p3,make_float3(cubeMin.x,cubeMin.y,cubeMin.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z)) ||
							rayTriangleIntersect(p1,p3,make_float3(cubeMin.x,cubeMin.y,cubeMin.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z)) ||
							rayTriangleIntersect(p1,p3,make_float3(cubeMin.x,cubeMin.y,cubeMin.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z)) ||
							rayTriangleIntersect(p1,p3,make_float3(cubeMin.x,cubeMax.y,cubeMax.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z)) ||
							rayTriangleIntersect(p1,p3,make_float3(cubeMin.x,cubeMax.y,cubeMax.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z)) ||
							rayTriangleIntersect(p1,p3,make_float3(cubeMin.x,cubeMax.y,cubeMax.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z)) ||
							rayTriangleIntersect(p1,p3,make_float3(cubeMax.x,cubeMax.y,cubeMin.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z)) ||
							rayTriangleIntersect(p1,p3,make_float3(cubeMax.x,cubeMax.y,cubeMin.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z)) ||
							rayTriangleIntersect(p1,p3,make_float3(cubeMax.x,cubeMax.y,cubeMin.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z)) ||
							rayTriangleIntersect(p1,p3,make_float3(cubeMax.x,cubeMin.y,cubeMax.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z)) ||
							rayTriangleIntersect(p1,p3,make_float3(cubeMax.x,cubeMin.y,cubeMax.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z)) ||
							rayTriangleIntersect(p1,p3,make_float3(cubeMax.x,cubeMin.y,cubeMax.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z)) ||
							// Repeat for triangle edge from p2 to p3
							rayTriangleIntersect(p2,p3,make_float3(cubeMin.x,cubeMin.y,cubeMin.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z)) ||
							rayTriangleIntersect(p2,p3,make_float3(cubeMin.x,cubeMin.y,cubeMin.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z)) ||
							rayTriangleIntersect(p2,p3,make_float3(cubeMin.x,cubeMin.y,cubeMin.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z)) ||
							rayTriangleIntersect(p2,p3,make_float3(cubeMin.x,cubeMax.y,cubeMax.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z)) ||
							rayTriangleIntersect(p2,p3,make_float3(cubeMin.x,cubeMax.y,cubeMax.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z)) ||
							rayTriangleIntersect(p2,p3,make_float3(cubeMin.x,cubeMax.y,cubeMax.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z)) ||
							rayTriangleIntersect(p2,p3,make_float3(cubeMax.x,cubeMax.y,cubeMin.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z)) ||
							rayTriangleIntersect(p2,p3,make_float3(cubeMax.x,cubeMax.y,cubeMin.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z)) ||
							rayTriangleIntersect(p2,p3,make_float3(cubeMax.x,cubeMax.y,cubeMin.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z),make_float3(cubeMin.x,cubeMax.y,cubeMin.z)) ||
							rayTriangleIntersect(p2,p3,make_float3(cubeMax.x,cubeMin.y,cubeMax.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z)) ||
							rayTriangleIntersect(p2,p3,make_float3(cubeMax.x,cubeMin.y,cubeMax.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z),make_float3(cubeMax.x,cubeMax.y,cubeMax.z)) ||
							rayTriangleIntersect(p2,p3,make_float3(cubeMax.x,cubeMin.y,cubeMax.z),make_float3(cubeMin.x,cubeMin.y,cubeMax.z),make_float3(cubeMax.x,cubeMin.y,cubeMin.z)) ){
								interSection = true;
						}

						if (interSection == true){

							//if (m_hCubeCounter[cubeIndex]+1 > m_maxTrianglesPerCube){
							//	printf("Too many triangles (%u) in cube %u!\n", m_hCubeCounter[cubeIndex]+1, cubeIndex);
							//	//std::cin.get();
							//}

							//m_hTrianglesInCubes[cubeIndex*m_maxTrianglesPerCube + m_hCubeCounter[cubeIndex]] = nTri;
							//m_hCubeCounter[cubeIndex] = m_hCubeCounter[cubeIndex] + 1;

							if (!determinedMaxTriPerCube){
								if (m_hCubeCounter[cubeIndex]+1 > m_maxTrianglesPerCube){
									m_maxTrianglesPerCube = m_hCubeCounter[cubeIndex] + 1;
								}
								m_hCubeCounter[cubeIndex] = m_hCubeCounter[cubeIndex] + 1;
							} else {
								m_hTrianglesInCubes[cubeIndex*m_maxTrianglesPerCube + m_hCubeCounter[cubeIndex]] = nTri;
								m_hCubeCounter[cubeIndex] = m_hCubeCounter[cubeIndex] + 1;
							}
						}
					}
				}
			}
		}

		printf("m_maxTrianglesPerCube: %u\n", m_maxTrianglesPerCube);
		printf("determinedMaxTriPerCube: %u\n", determinedMaxTriPerCube);

		if (!determinedMaxTriPerCube){
			determinedMaxTriPerCube = true;
		} else {
			populatedCubes = true;
		}
	}

	//std::cin.get();
	
	// Count maximum and average number of triangles per cube (for testing purposes)
	uint secondMaxCountedTriangles = 0;
	uint maxCountedTriangles = 0;
	float avTrianglesInCubes = 0;
	uint i_max = 0;
	for (uint i=0; i<m_totalNumCubes; i++){
		avTrianglesInCubes += m_hCubeCounter[i];
		if (m_hCubeCounter[i] > maxCountedTriangles){
			secondMaxCountedTriangles = maxCountedTriangles;
			maxCountedTriangles = m_hCubeCounter[i];
			i_max = i;
		}
	}
	avTrianglesInCubes = avTrianglesInCubes/((float) m_totalNumCubes);

	uint counter[maxCountedTriangles+1];
	memset(counter,0,(maxCountedTriangles+1)*sizeof(uint));
	for (uint i=0; i<m_totalNumCubes; i++){
		counter[m_hCubeCounter[i]]++;
	}

	for (uint i=0; i<maxCountedTriangles+1; i++){
		printf("Number of cubes having %u triangles: %u\n", i, counter[i]);
	}


	printf("Average triangles per cube: %g\n", avTrianglesInCubes);
	printf("Maximum triangle count: %u\n", maxCountedTriangles);
	printf("Second biggest triangle count: %u\n", secondMaxCountedTriangles);

	printf("m_totalNumCubes: %u\n", m_totalNumCubes);
	printf("m_hCubeCounter[%u]: %u\n", i_max, m_hCubeCounter[i_max]);
	printf("m_hTrianglesInCubes for cube %u: ", i_max);
	for (uint i=0; i<maxCountedTriangles; i++){
		printf("%u,", m_hTrianglesInCubes[i_max*m_maxTrianglesPerCube + i]);
	}
	printf("\n");
	// Done with testing

	if (m_useGpu){
		///////////////////////////////////////////////////////////////////////////////////////
		// Copy cube info (m_numCubes, m_totalNumCubes, m_maxTrianglesPerCube, m_cubeLength, 
		// m_hTrianglesInCubes and m_hCubeCounter to device memory
		///////////////////////////////////////////////////////////////////////////////////////
		printf("Copying cube info (m_numCubes, m_totalNumCubes, m_maxTrianglesPerCube, m_cubeLength, m_hTrianglesInCubes and m_hCubeCounter to device memory\n");

		allocateArray((void**)&m_dTrianglesInCubes, m_totalNumCubes*m_maxTrianglesPerCube*sizeof(uint));
		allocateArray((void**)&m_dCubeCounter, m_totalNumCubes*sizeof(uint));

		copyArrayToDevice(m_dTrianglesInCubes, m_hTrianglesInCubes, 0, m_totalNumCubes*m_maxTrianglesPerCube*sizeof(uint));
		copyArrayToDevice(m_dCubeCounter, m_hCubeCounter, 0, m_totalNumCubes*sizeof(uint));

		char gpuName_numCubes [] = "k_numCubes";
		copyConstantToDevice(gpuName_numCubes, &m_numCubes, 0, sizeof(uint));

		char gpuName_totalNumCubes [] = "k_totalNumCubes";
		copyConstantToDevice(gpuName_totalNumCubes, &m_totalNumCubes, 0, sizeof(uint));

		char gpuName_maxTrianglesPerCube [] = "k_maxTrianglesPerCube";
		copyConstantToDevice(gpuName_maxTrianglesPerCube, &m_maxTrianglesPerCube, 0, sizeof(uint));

		char gpuName_cubeLength [] = "k_cubeLength";
		copyConstantToDevice(gpuName_cubeLength, &m_cubeLength, 0, sizeof(float));

		printf("Done copying cube info to device memory\n");
	}
}


bool SpinSystem::pointInCube(float3 p, float3 cubeMin, float3 cubeMax){
	return ( ((p.x <= cubeMax.x)&&(p.x >= cubeMin.x)) && ((p.y <= cubeMax.y)&&(p.y >= cubeMin.y)) && ((p.z <= cubeMax.z)&&(p.z >= cubeMin.z)) );
}


bool SpinSystem::rayTriangleIntersect(float3 rayPoint1, float3 rayPoint2, float3 triPoint1, float3 triPoint2, float3 triPoint3){
	float3 u = triPoint2-triPoint1;
	float3 v = triPoint3-triPoint1;
	float3 n = make_float3(u.y*v.z-u.z*v.y,u.z*v.x-u.x*v.z,u.x*v.y-u.y*v.x);
	float nlength = sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
	float s = 2.0f, t = 2.0f;
	n.x = n.x/nlength;
	n.y = n.y/nlength;
	n.z = n.z/nlength;

	float r = dot(n,triPoint1-rayPoint1)/dot(n,rayPoint2-rayPoint1);

	if ((0<r)&(r<1)){
		float3 w = rayPoint1 + r*(rayPoint2-rayPoint1) - triPoint1;
		s = (dot(u,v)*dot(w,v)-dot(v,v)*dot(w,u))/(dot(u,v)*dot(u,v)-dot(u,u)*dot(v,v));
		t = (dot(u,v)*dot(w,u)-dot(u,u)*dot(w,v))/(dot(u,v)*dot(u,v)-dot(u,u)*dot(v,v));
	}

	return ( (s>=0) && (t>=0) && (s+t<=1) );
}


//////////////////////////////////////////////////////////////////////////////
// Function name:	resetSpins
// Description:		Reset spin positions (to random values) and signal
//////////////////////////////////////////////////////////////////////////////		Note: Need to make sure that findSpinsInFibers is applied correctly afterwards
void SpinSystem::resetSpins(){

	for (uint i=0; i<m_numSpins; i++){
		m_hPos[i*m_nPosValues] = m_startBoxSize*2*(frand() - 0.5f);		// x position
		m_hPos[i*m_nPosValues+1] = m_startBoxSize*2*(frand() - 0.5f);		// y position
		m_hPos[i*m_nPosValues+2] = m_startBoxSize*2*(frand() - 0.5f);		// z position
		
		//m_hSpins[i*m_nSpinValues+1] = 1.0f;			// Signal magnitude
		//m_hSpins[i*m_nSpinValues+2] = 0.0f;			// Signal phase
		m_hSpins[i].signalMagnitude = 1.0f;
		m_hSpins[i].signalPhase = 0.0f;
	}
	
	// Find which compartment/fiber each spin belongs to
	printf("Assigning spins to fibers\n");
	findSpinsInFibers();
	printf("Done assigning spins to fibers\n");

	assignSeeds();
	setArray();
	setSpinArray();
}


////////////////////////////////////////////////////////////////////////////////
// Function name: getMrSignal
// Description: Get the total signal from all the particles.
// 		If no input arguments, gives signal for all compartments.
//		If array is passed as parameter, writes the signal contributions
//		from all compartments into the array, and the total signal
//		in the last element of the array
////////////////////////////////////////////////////////////////////////////////
double SpinSystem::getMrSignal(){
	//printf("signalMagnitude[10] (1): %g \n", m_hSpins[10].signalMagnitude);
	getArray();
	getSpinArray();						// Test
	
	//printf("signalMagnitude[10] (2): %g \n", m_hSpins[10].signalMagnitude);
	double xMagn = 0, yMagn = 0, mrSignal;
	//float *posPtr = m_hPos;
	//float *spinPtr = m_hSpins;
	/////////////////////////////////////////////////////////////////////////
	// Only measure spins in the center to avoid edge effects. d specifies
	// the proportion of the voxel to measure. E.g., d = 0.8 will measure all
	// spins that are within +/-0.8 from the center of the (-1 to +1) voxel.
	/////////////////////////////////////////////////////////////////////////
	//double d = 0.80f;
	uint n = 0;
	for (uint i=0; i<m_numSpins; i++){
		//if (posPtr[0] > -d && posPtr[0] < d && posPtr[1] > -d && posPtr[1] < d && posPtr[2] > -d && posPtr[2] < d){
		//if (spinPtr[1] > 0){
			// Calculate the total x- and y-components
			//xMagn += cos(spinPtr[2])*spinPtr[1];
			//yMagn += sin(spinPtr[2])*spinPtr[1];
			xMagn += cos(m_hSpins[i].signalPhase)*m_hSpins[i].signalMagnitude;
			yMagn += sin(m_hSpins[i].signalPhase)*m_hSpins[i].signalMagnitude;
			n++;
		//}
		//posPtr += m_nPosValues;
		//spinPtr += m_nSpinValues;
	}
	// We calculate the average signal magnitude from the x- and y-components
	mrSignal = sqrt(xMagn*xMagn+yMagn*yMagn)/(double)n;
	return mrSignal;
}

void SpinSystem::getMrSignal(double mrSignal[]){
	getArray();
	getSpinArray();						// Test
	
	double xMagn[m_numCompartments+1], yMagn[m_numCompartments+1];
	memset(xMagn,0,(m_numCompartments+1)*sizeof(double));
	memset(yMagn,0,(m_numCompartments+1)*sizeof(double));
	uint n[m_numCompartments+1];
	memset(n,0,(m_numCompartments+1)*sizeof(uint));
	//float *posPtr = m_hPos;
	//float *spinPtr = m_hSpins;
	/////////////////////////////////////////////////////////////////////////
	// Only measure spins in the center to avoid edge effects. d specifies
	// the proportion of the voxel to measure. E.g., d = 0.8 will measure all
	// spins that are within +/-0.8 from the center of the (-1 to +1) voxel.
	/////////////////////////////////////////////////////////////////////////
	//double d = 0.80f;
	
	for (uint i=0; i<m_numSpins; i++){
		//if (posPtr[0] > -d && posPtr[0] < d && posPtr[1] > -d && posPtr[1] < d && posPtr[2] > -d && posPtr[2] < d){
		//if (spinPtr[1] > 0){
			// Calculate the total x- and y-components
			// uint compartment = spinPtr[0];
			uint compartment = m_hSpins[i].compartmentType;
			//xMagn[0] += cos(spinPtr[2])*spinPtr[1];
			//yMagn[0] += sin(spinPtr[2])*spinPtr[1];
			xMagn[0] += cos(m_hSpins[i].signalPhase)*m_hSpins[i].signalMagnitude;
			yMagn[0] += sin(m_hSpins[i].signalPhase)*m_hSpins[i].signalMagnitude;
			n[0]++;

			//xMagn[compartment+1] += cos(spinPtr[2])*spinPtr[1];
			//yMagn[compartment+1] += sin(spinPtr[2])*spinPtr[1];
			xMagn[compartment+1] += cos(m_hSpins[i].signalPhase)*m_hSpins[i].signalMagnitude;
			yMagn[compartment+1] += sin(m_hSpins[i].signalPhase)*m_hSpins[i].signalMagnitude;
			n[compartment+1]++;
		//}
		//posPtr += m_nPosValues;
		//spinPtr += m_nSpinValues;
	}
	// We calculate the average signal magnitude from the x- and y-components
	for (uint i=0; i<m_numCompartments+1; i++){
		printf("n[%u]: %u\n", i, n[i]);
		if (n[i] == 0){
			mrSignal[i] = 0;
		} else {
			mrSignal[i] = sqrt(xMagn[i]*xMagn[i]+yMagn[i]*yMagn[i])/(double)n[i];
		}
	}
}


//////////////////////////////////////////////////////////////////////////////////////////
// Function name:	setColorFromSignal
// Description:		Sets the color of the particle based on its signal phase
//////////////////////////////////////////////////////////////////////////////////////////
void SpinSystem::setColorFromSignal(){
	getArray();
	// set color buffer
	glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
	float *colPtr = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	float magnitude;
	for (uint i=0; i<m_numSpins; i++){
		// Set the color of the particle based on the signal magnitude and phase.
		//magnitude = m_hSpins[i].signalMagnitude;
		//*colPtr++ = (0.5 + sin(m_hSpins[i].signalPhase)/2)*magnitude;
		//*colPtr++ = 0.5*magnitude;
		//*colPtr++ = 0.5*magnitude;
		*colPtr++ = (m_hSpins[i].compartmentType == 0);
		*colPtr++ = (m_hSpins[i].compartmentType == 2);
		*colPtr++ = (m_hSpins[i].compartmentType == 1);	
	}
	glUnmapBufferARB(GL_ARRAY_BUFFER);
}


//////////////////////////////////////////////////////////////////////////
// Function name:	assignSeeds
// Description:		Give the elements of m_hSeed (later used for random 
//			walk calculations) random values in the range of uint
//////////////////////////////////////////////////////////////////////////
void SpinSystem::assignSeeds(){
	for (uint i=0; i<m_numSpins; i++){
		m_hSeed[i*m_nSeedValues] = rand()*(UINT_MAX/RAND_MAX);
		m_hSeed[i*m_nSeedValues+1] = rand()*(UINT_MAX/RAND_MAX);
	}
}


///////////////////////////////////////////////////////////////////////////
// Function name:	getArray
// Description:		Get the position array from device to host
///////////////////////////////////////////////////////////////////////////
float* SpinSystem::getArray(){
	float *hdata = m_hPos;
	float *ddata = m_dPos[m_currentPosRead];

	if (m_useDisplay){
		unsigned int vbo = m_posVbo[m_currentPosRead];
		copyArrayFromDevice(hdata, ddata, vbo, m_numSpins*m_nPosValues*sizeof(float));
	} else{
		copyArrayFromDevice(hdata, ddata, 0, m_numSpins*m_nPosValues*sizeof(float));
	}

	return hdata;
}

/////////////////////////////////////////////////////////////////////////////
// Function name:	getSpinArray
// Description:		Get the spin information array from device to host.
//			Note: Should maybe combine with getArray
/////////////////////////////////////////////////////////////////////////////
//float* SpinSystem::getSpinArray(){
	//float* hSpinData = m_hSpins;										// Test
	//float *dSpinData = m_dSpins[m_currentSpinRead];								// Test
	//copyArrayFromDevice(hSpinData, dSpinData, 0, m_numSpins*m_nSpinValues*sizeof(float));			// Test


spinData* SpinSystem::getSpinArray(){

	spinData* hSpinData = m_hSpins;
	spinData* dSpinData = m_dSpins[m_currentSpinRead];
	copyArrayFromDevice(hSpinData, dSpinData, 0, m_numSpins*sizeof(spinData));

	return hSpinData;
}


////////////////////////////////////////////////////////////////////////
// Function name:	setArray
// Description:		Copy the position and seed arrays from host to
//			device.
////////////////////////////////////////////////////////////////////////
void SpinSystem::setArray(){
	// Copy the position array from the host machine to the GPU
	if (m_useDisplay){
		unregisterGLBufferObject(m_posVbo[m_currentPosRead]);
		glBindBuffer(GL_ARRAY_BUFFER, m_posVbo[m_currentPosRead]);
		glBufferSubData(GL_ARRAY_BUFFER, 0, m_numSpins*m_nPosValues*sizeof(float), m_hPos);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		registerGLBufferObject(m_posVbo[m_currentPosRead]);
	} else{
		copyArrayToDevice(m_dPos[m_currentPosRead], m_hPos, 0, m_numSpins*m_nPosValues*sizeof(float));
	}
	// Copy the seed array from the host machine to the GPU
	copyArrayToDevice(m_dSeed[m_currentSeedRead], m_hSeed, 0, m_numSpins*m_nSeedValues*sizeof(uint));
}

///////////////////////////////////////////////////////////////////////////
// Function name:	setSpinArray
// Description:		Copy the spin information array from host to device.
///////////////////////////////////////////////////////////////////////////
void SpinSystem::setSpinArray(){
	//copyArrayToDevice(m_dSpins[m_currentSpinRead], m_hSpins, 0, m_numSpins*m_nSpinValues*sizeof(float));
	copyArrayToDevice(m_dSpins[m_currentSpinRead], m_hSpins, 0, m_numSpins*sizeof(spinData));
}

#define HITARRAYSIZE 2000
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function name:	findSpinsInFibers
// Description:		Assign each spin to a compartment: Compartment no. 0 is the axon or glia, no. 1 is the myelin 
//			and no. 2 is the outside of the fiber. The compartment number is stored in 
//			m_hSpins[spinIndex*m_nSpinValues]
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SpinSystem::findSpinsInFibers(){
	
	// We time the procedure - will delete soon
	time_t start, end;
	time(&start);

	collResult tempResult;
	float3 spinPos, lineEndPos;
	float minxyz[3], maxxyz[3];
	uint hitArray[HITARRAYSIZE], hitArrayCurrentMembrane[HITARRAYSIZE];
	uint spinCompartment, triIndex, triIndex2, fiberIndex, fiberIndex2, membraneIndex, membraneIndex2, triInd, nHitsCurrentMembrane, fiberInside;
	uint *compartmentCounter = new uint[m_numCompartments];
	memset(compartmentCounter, 0, m_numCompartments*sizeof(uint));
	uint maxTriesPerSpin = 20;

	bool printTest = 0;

	////////////////////////////////////////////////////////////////////
	// The method for determining the compartment is as follows:
	//	For each particle, we draw a line from the particle to the outside
	//	of the box. We then look at how many times the line intersects fiber
	//	membranes. If the line intersects a membrane belonging to a particular
	//	fiber an odd number of times, then it is enclosed by that membrane.
	//	We first check the innermost membranes and work our way outwards.
	//	This requires all membranes to be closed surfaces, but does not
	//	require them to be convex.
	////////////////////////////////////////////////////////////////////
	uint16 spinIndex=0;
	uint tryCounter = 0;
	uint nHitsMax = 0;

	//m_hPos[0] = 0.075;
	//m_hPos[1] = 0.07;
	

	while (spinIndex < m_numSpins){

		//if (spinIndex == 10){
		//	printTest = 1;
		//}

		uint nEdgeCollisions = 0;
		uint nVertexCollisions = 0;

		// We have tried determining the compartment of this spin maxTriesPerSpin times without success (the spin is probably located
		// on a triangle edge). We just move the spin to a new random position.
		if (tryCounter > maxTriesPerSpin){
			m_hPos[spinIndex*m_nPosValues] = m_startBoxSize*2*(frand() - 0.5f);		// x position
			m_hPos[spinIndex*m_nPosValues+1] = m_startBoxSize*2*(frand() - 0.5f);		// y position
			m_hPos[spinIndex*m_nPosValues+2] = m_startBoxSize*2*(frand() - 0.5f);		// z position
			tryCounter = 0;
		}
		
		spinPos.x = m_hPos[spinIndex*m_nPosValues];
		spinPos.y = m_hPos[spinIndex*m_nPosValues+1];
		spinPos.z = m_hPos[spinIndex*m_nPosValues+2];		

		
		// lineEndPos is the endpoint of a line that extends from the spin to the outside of the volume
		// We create the shortest line possible.
		lineEndPos = spinPos;

		if ( (fabs(spinPos.x)>fabs(spinPos.y)) & (fabs(spinPos.x)>fabs(spinPos.z)) ){		// The shortest line is along the x-axis
			lineEndPos.x = ((spinPos.x>=0)-(spinPos.x<0))*1.1*m_xmax;				// (u>0)-(u<0) = sign(u)
		} else if (fabs(spinPos.y)>fabs(spinPos.z)){						// The shortest line is along the y-axis
			lineEndPos.y = ((spinPos.y>=0)-(spinPos.y<0))*1.1*m_ymax;
		} else {										// The shortest line is along the z-axis
			lineEndPos.z = ((spinPos.z>=0)-(spinPos.z<0))*1.1*m_zmax;
		}
		//printf("LineEndPos: [%g,%g,%g]\n", lineEndPos.x, lineEndPos.y, lineEndPos.z);

		// If the "tryCounter" is above zero, that means that we previously looked at this same spin and drew a line which intersected an edge or a
		// vertex of a triangle, which made the result ambiguous. We therefore look at the same spin again and tilt the line a bit in each direction
		// (i.e. try a new line and hope that we don't intersect a triangle edge/vertex.
		if (tryCounter > 0){
			printf("Trial no. %u for spin no. %u\n", tryCounter, spinIndex);
			lineEndPos.x = lineEndPos.x + frand()*0.05;
			lineEndPos.y = lineEndPos.y + frand()*0.05;
			lineEndPos.z = lineEndPos.z + frand()*0.05;
		}

		// Finding minx, miny, minz
		minxyz[0] = spinPos.x; if (lineEndPos.x < minxyz[0]){minxyz[0] = lineEndPos.x;}
		minxyz[1] = spinPos.y; if (lineEndPos.y < minxyz[1]){minxyz[1] = lineEndPos.y;}
		minxyz[2] = spinPos.z; if (lineEndPos.z < minxyz[2]){minxyz[2] = lineEndPos.z;}
	
		// Finding maxx, maxy, maxz
		maxxyz[0] = spinPos.x; if (lineEndPos.x > maxxyz[0]){maxxyz[0] = lineEndPos.x;}
		maxxyz[1] = spinPos.y; if (lineEndPos.y > maxxyz[1]){maxxyz[1] = lineEndPos.y;}
		maxxyz[2] = spinPos.z; if (lineEndPos.z > maxxyz[2]){maxxyz[2] = lineEndPos.z;}

		// Find all collisions between our line and the triangle mesh
		int nHits = totalTree.Search(minxyz, maxxyz, hitArray);

		if (nHits > nHitsMax){ nHitsMax = nHits; }
		
		///////////////////////////////////////////////////////////////////////////////////
		// If nHits (number of collisions) exceeds the size of hitArray (which stores the indices
		// of the collision triangles) we get an error. Need to make hitArray big enough or of
		// dynamic size.
		///////////////////////////////////////////////////////////////////////////////////
		if (nHits>HITARRAYSIZE){ 
			printf("spinSystem.cpp: nHits>HITARRAYSIZE, about to segfault! Edit me and adjust HITARRAYSIZE.\n");
			printf("spin index: %u\n", spinIndex);
			printf("nHits: %u\n", nHits);
			printf("spinPos: [%g,%g,%g]\n", spinPos.x, spinPos.y, spinPos.z);
			printf("lineEndPos: [%g,%g,%g]\n", lineEndPos.x, lineEndPos.y, lineEndPos.z);
			printf("minxyz: [%g,%g,%g]\n", minxyz[0],minxyz[1],minxyz[2]);
			printf("maxxyz: [%g,%g,%g]\n", maxxyz[0],maxxyz[1],maxxyz[2]);
			printf("Trycounter: %u\n", tryCounter);
			
			for (uint p=0; p<nHits; p++){
				printf("hitArray[%u]: %u\n", p, hitArray[p]);
			}
			//printTest = 1;
			//std::cin.get();
		}

		uint membraneType = 0;										// We start with membrane type no. 0 (axon surface)
		fiberInside = UINT16_MAX;										// fiberInside is the index of the fiber which contains the particle.
														// If particle is outside all fibers then fiberInside = UINT_MAX
		// Loop through membrane types
		while ((membraneType<m_nMembraneTypes)&&(fiberInside==UINT16_MAX)){					
			for (uint k=0; k<nHits; k++){								// We loop through the triangles which the line possibly intersects
				triIndex = hitArray[k];								// according to the R-Tree
				if (triIndex<UINT_MAX){
					fiberIndex = m_hTriInfo[triIndex*3];				// Find the fiber which the triangle belongs to
					membraneIndex = m_hTriInfo[triIndex*3+1];			// Find which membrane type the triangle belongs to
					if (membraneIndex == membraneType){					// The triangle belongs to the membrane type being examined
						hitArray[k] = UINT_MAX;					// Set the triangle in the collision array to UINT_MAX to avoid multiple checks
						nHitsCurrentMembrane = 1;					// We have found one collision with the membrane being examined
						hitArrayCurrentMembrane[nHitsCurrentMembrane-1] = triIndex;	// Store the triangle in hitArrayCurrentMembrane, which
			
						for (uint q = k+1; q<nHits; q++){		// Check how many of the remaining triangles in hitArray belong to the same membrane
							triIndex2 = hitArray[q];
							if (triIndex2<UINT_MAX){
								fiberIndex2 = m_hTriInfo[triIndex2*3];
								membraneIndex2 = m_hTriInfo[triIndex2*3+1];
			
								if ((fiberIndex2==fiberIndex)&(membraneIndex2==membraneIndex)){
									nHitsCurrentMembrane++;
									hitArrayCurrentMembrane[nHitsCurrentMembrane-1] = triIndex2;
									hitArray[q] = UINT_MAX;
								}
							}
						}
		

						/////////////////////////////////////////////////////////////////
						// We now have a collection of triangles each belonging to the 
						// same membrane, that the R-Tree detected for possible collisisons.
						// We now need to loop through them to determine whether actual collisions occur.
						////////////////////////////////////////////////////////////////
						uint nCollisions = 0;

						for (uint p=0;p<nHitsCurrentMembrane;p++){
							triInd = hitArrayCurrentMembrane[p];
							tempResult = triCollDetect(spinPos, lineEndPos, triInd);
							if (tempResult.collisionType>0){			// collisionType > 0: We have collision
									//printf("Collision with triangle %u!\n", triInd);
								nCollisions++;
								if (tempResult.collisionType == 2){		// collisionType == 2: Collision with edge of triangle
									//printf("Edge collision!\n");
									//printf("SpinIndex: %u\n", spinIndex);
									//tryAgain = true;
									printf("Edge collision with triangle %u\n", tempResult.collIndex);
									nEdgeCollisions++;
								}
								if (tempResult.collisionType == 3){		// collisionType == 3: Collision with vertex of triangle - should
									//printf("Vertex collision!\n");		// be extremely rare
									//printf("SpinIndex: %u\n", spinIndex);
									//tryAgain = true;									
									////std::cin.get();
									nVertexCollisions++;
								}
							}
						}
						//nCollisions -= nEdgeCollisions/2;				// For each edge collision, we have two triangle collisions which
														// should only count as one membrane collision - subtract half of them
		
						if (nCollisions % 2 == 1){					// We have odd number of collisions with current membrane, so particle
							fiberInside = fiberIndex;				// is inside that membrane.
							if (membraneType == 0){
								spinCompartment = 1;
							} else if (membraneType == 2){
								spinCompartment = 3;
							} else {
								spinCompartment = 2;
							}
						}
					}
				}
			}
			membraneType++;
		}

		if (fiberInside == UINT16_MAX){									// Particle is outside all membranes.
			spinCompartment = 0;									// Note: Am assuming that compartment 0 is outside all fibers
			//printf("Test4\n");
		}

		if ( (nEdgeCollisions == 0) && (nVertexCollisions == 0) ){
			m_hSpins[spinIndex].compartmentType = spinCompartment;
			m_hSpins[spinIndex].insideFiber = fiberInside;
			compartmentCounter[spinCompartment]++;
			spinIndex++;
			tryCounter = 0;
		} else {
			printf("%u edge collisions and %u vertex collisions for spin %u at [%g,%g,%g]\n", nEdgeCollisions, nVertexCollisions, spinIndex, spinPos.x, spinPos.y, spinPos.z);
			printf("Trying again\n");
			tryCounter++;
		}

		if (printTest){
			printf("Spin index: %u\n", spinIndex-1);
			printf("Spin position: [%g,%g,%g]\n", spinPos.x, spinPos.y, spinPos.z);
			printf("compartment: %u\n", m_hSpins[spinIndex-1].compartmentType);
			printf("fiberInside: %u\n", m_hSpins[spinIndex-1].insideFiber);
			//std::cin.get();
		}
	}



	printf("(in SpinSystem::findSpinsInFibers):");
	for (uint c=0; c<m_numCompartments; c++){
		printf("compartmentCounter[%u]: %u, ", c, compartmentCounter[c]);
	}	
	printf("\n");

	printf("Max no. of hits: %u\n", nHitsMax);

	delete [] compartmentCounter;

	time(&end);
	double dif = difftime(end,start);
	printf("(in SpinSystem::findSpinsInFibers): Checking particles took %g seconds.\n", dif);
}


//////////////////////////////////////////////////////////////////////////
// Function name:	getNumTriInMembrane
// Description:		Get the number of triangles in a particular 
//			membrane type
//////////////////////////////////////////////////////////////////////////
uint SpinSystem::getNumTriInMembraneType(uint membraneType){
	return m_nTrianglesInMembraneType[membraneType];
}


//////////////////////////////////////////////////////////////////////////
// Function name:	getTriPoint
// Description:		Get coordinates of point number 0, 1 or 2 in a 
//			particular triangle
//////////////////////////////////////////////////////////////////////////
float3 SpinSystem::getTriPoint(uint nTri,uint nPoint){
	
	float3 returnPoint;
	uint pointIndex = m_trgls[nTri*3+nPoint];
	returnPoint.x = m_vertices[pointIndex*3+0];
	returnPoint.y = m_vertices[pointIndex*3+1];
	returnPoint.z = m_vertices[pointIndex*3+2];

	return returnPoint;
}


//////////////////////////////////////////////////////////////////////////
// Function name:	getNumFibers
// Description:		Returns the number of fibers in the volume
//////////////////////////////////////////////////////////////////////////
uint SpinSystem::getNumFibers(){
	return m_nFibers;
}


//////////////////////////////////////////////////////////////////////////
// Function name:	getNumCompartments
// Description:		Returns the number of compartment types in the volume
//////////////////////////////////////////////////////////////////////////
uint SpinSystem::getNumCompartments(){
	return m_numCompartments;
}


//////////////////////////////////////////////////////////////////////////
// Function name:	getNumTriInFiberMembrane
// Description:		Returns the number of triangles belonging to a
//			particular fiber and membrane type
//////////////////////////////////////////////////////////////////////////
uint SpinSystem::getNumTriInFiberMembrane(uint membraneType, uint fiberNr){
	return m_triCounter[membraneType*m_nFibers+fiberNr];
}


//////////////////////////////////////////////////////////////////////////
// Function name:	getTriInFiberArray
// Description:		Returns the index of triangle number triIndex, belonging
//			to fiber fiberIndex and membrane type membraneIndex
//////////////////////////////////////////////////////////////////////////
uint SpinSystem::getTriInFiberArray(uint membraneIndex, uint fiberIndex, uint triIndex){
	return m_fibers[(membraneIndex*m_nFibers+fiberIndex)*m_maxTrianglesOnSurface+triIndex];
}


///////////////////////////////////////////////////////////////////////////
// Function name:	updateSpins
// Description:		Update the spins position by calling the spin 
//			kernel for each iteration. Also scale spatial 
//			parameters to fit our space.
///////////////////////////////////////////////////////////////////////////
float SpinSystem::updateSpins(float deltaTime, uint iterations){
	float phaseConstant;
	float3 gradScaled;

	// Convert mT/m to T/um and then scale to our space by multiplying with spaceScale
	gradScaled = m_gradient * 1e-9f * m_spaceScale;

	// The expected phase shift is dot(G,pos) * 2*pi * gyromagneticRatio * deltaTime.
	// The dot-product will be computed in the spin kernel for each unique position.
	// We compute the rest out here for efficiency.
	// The gyromagnetic ratio is in KHz/T, which is equivalent to (cycles/millisecond)/T
	// Since our time units are ms, this means that we don't need to scale.
	phaseConstant = TWOPI * m_gyroMagneticRatio * deltaTime;

	//printf("m_hSeed: [%u,%u]\n", m_hSeed[0], m_hSeed[1]);

	if (m_useGpu){
		if(m_useDisplay){
            		integrateSystemVBO( m_posVbo[m_currentPosRead],
                	                m_dSeed[m_currentSeedRead],
					m_dSpins[m_currentSpinRead],
                	                deltaTime,
                	                m_permeability,
                	                m_numSpins,
                	                gradScaled,
                	                phaseConstant,
                	                iterations, m_dTrianglesInCubes, m_dCubeCounter );
        	} else {
            		integrateSystem(    m_dPos[m_currentPosRead],
                	                m_dSeed[m_currentSeedRead], 
					m_dSpins[m_currentSpinRead],
                	                deltaTime,
                	                m_permeability,
                	                m_numSpins,
                	                gradScaled,
                	                phaseConstant,
                	                iterations, m_dTrianglesInCubes, m_dCubeCounter );
        	}
	} else {
		for (uint i=0;i<iterations; i++){
			cpuIntegrateSystem(deltaTime, gradScaled, phaseConstant);
		}
	}

	return 0;
}

#include "spinKernelCpu.cpp"
