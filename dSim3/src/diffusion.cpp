//////////////////////////////////////////////////////////////////////////////////////////////////
// File name: diffusion.cpp
// Description: Main file for running diffusion simulator (dSim).
//////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <libconfig.h++>
#include <cutil.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut.h>
#include <math.h>
#include <cstdlib>
#include <cstdio>

#include <jama_svd.h>
#include <jama_eig.h>
#include <tnt_array2d.h>
#include <tnt_array1d.h>

#include "renderSpins.h"
#include "paramgl.h"
#include "spinSystem.h"

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
// Define global variables
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Define varibles for use in the graphical rendering of the simulation.
////////////////////////////////////////////////////////////////////////////////
int ox, oy;
int buttonState = 0;
float camera_trans[] = {0, 0, -3};
float camera_rot[] = {0, 0, 0};
float camera_trans_lag[] = {0, 0, -3};
float camera_rot_lag[] = {0, 0, 0};
const float inertia = 0.1;
float modelview[16];
float myelinAlpha = 0.3f;
float gliaAlpha = 0.5f;
bool bPause = false;
SpinRenderer::DisplayMode displayMode = SpinRenderer::SPIN_SPHERES;
SpinRenderer *renderer = 0;

//////////////////////////////////////////////////////////////////////////////
// Define a new data structure for storing the magnetic gradient sequence.
// The gradients will be organized as a linked list, each link of type gradStruct.
// lDelta: Little delta, width of the pulse
// bDelta: Big delta, time between consecutive pulses
// readOut: The readout time
// gx, gy, gz: Gradient components in x, y, z directions
// next: The pointer to the next link in the list
//////////////////////////////////////////////////////////////////////////////
typedef struct _gradStruct{
	float lDelta;
	float bDelta;
	float readOut;
	float gx;
	float gy;
	float gz;
	struct _gradStruct *next;
} gradStruct;

/////////////////////////////////////////////////////////////////////////////
// Define variable related to the computation and writing of the output.
//
// NB: Should try to make these local instead of global
/////////////////////////////////////////////////////////////////////////////
gradStruct *gGrads = NULL;			// Will point to the magnetic gradient sequence structure					
char *fiberFile = NULL;				// Will point to the name of the file containing the fiber triangle mesh
char *outFileName = NULL;			// Will point to the name of the output file
FILE *outFilePtr;				// 
SpinSystem *psystem = 0;			// Will point to our spin system object
uint stepCount = 0;				// Counts the number of computational steps performed (=updateCount*stepsPerUpdate)
uint updateCount = 0;				// Counts the number of display updates performed
float *gradientDataResults;				// Contains numOutputs output values for each gradient: 
uint numOutputs;				// For each gradient: lDelta, bDelta, readOut, gx, gy, gz and signal at the end of gradient run for each compartment and all compartments
uint numGrads = 0;				// Number of magnetic gradients in the sequence (normally 13, including a zero gradient in the beginning)
bool computedOutput = 0;			// Will be set to 1 once the output signal has been computed to form a tensor and its eigenvalues
float ell1, ell2, ell3, ex, ey, ez, theta;	// ell1,2,3: The lengths of the axes of the diffusion ellipsoid. ex,y,z: Components of principal eigenvector of ellipsoid.
						// Theta: The angle between the principal eigenvector and the x-axis.

/////////////////////////////////////////////////////////////////////////////
// Define default values for the parameters specified in the configuration file.
// Will be overwritten if a configuration file is specified.
/////////////////////////////////////////////////////////////////////////////
float timeStep = 0.01;				// ms
int stepsPerUpdate = 1;
bool useGpu = true;
bool useDisplay = true;
uint triSearchMethod = 0;
uint reflectionType = 1;
bool writeOut = true;
uint numSpins = 30000;
float gyroMagneticRatio = 42576.0f;		// kHz/T
float extraAdc = 2.1;				// um^2/s
float intraAdc = 2.1;
float myelinAdc = 0.1;
float spaceScale = 100.0f;
float permeability = 0.000002;
float extraT2 = 79.0;
float intraT2 = 79.0;
float myelinT2 = 7.6;
float startBoxSize = 0.6;


////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
// Done with defining global parameters, start defining functions.
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
// Function name: readInputs.
// Description: Reads parameters from the configuration file, overwrites their default 
//		values.
// Inputs:	Name of configuration file plus all parameters in config file (passed by 
//		reference).
//////////////////////////////////////////////////////////////////////////////////////////
void readInputs(int argc, char** argv, char *configFilename, int *pstepsPerUpdate, bool *puseGpu, bool *puseDisplay, bool *pwriteOut, uint *psearchMethod, uint *pnumSpins, 
		float *pgyroMagneticRatio, float *ptimeStep, float *pextraAdc, float *pintraAdc, float *pmyelinAdc, float *pextraT2, float *pintraT2, 
		float *pmyelinT2, float *pspaceScale, float *ppermeability, float *pstartBoxSize, char *& outFileName){
	using namespace libconfig;
	using namespace std;

	////////////////////////////////////////////////////////////////////////
	// Note: Usage of cfg commands gives errors of type "Undefined reference to..." if -lconfig++ flag
	// is omitted from the compile command in the Makefile.
	// Errors in syntax in the sim.cfg file (such as omitted ;) can result in parse errors.
	////////////////////////////////////////////////////////////////////////
	Config cfg;
	try{
		cout << "Loading " << configFilename << "..." << endl;
		cfg.readFile(configFilename);
	} catch(FileIOException){
		cout << "File IO failed." << endl;
	} catch(ParseException){
		cout << "Parse failed." << endl;
	}

	//////////////////////////////////////////////////////////////////////
	// Read the app parameters from the cfg file
	//////////////////////////////////////////////////////////////////////
	cfg.lookupValue("app.stepsPerUpdate", *pstepsPerUpdate);
	cfg.lookupValue("app.useGpu", *puseGpu);
	cfg.lookupValue("app.useDisplay", *puseDisplay);
	cfg.lookupValue("app.writeOut", *pwriteOut);

	//////////////////////////////////////////////////////////////////////
	// Read the sim parameters from the cfg file
	//////////////////////////////////////////////////////////////////////
	cfg.lookupValue("sim.numSpins", *pnumSpins);
	cfg.lookupValue("sim.gyroMagneticRatio", *pgyroMagneticRatio);
	cfg.lookupValue("sim.timeStep", *ptimeStep);
	cfg.lookupValue("sim.extraAdc", *pextraAdc);
	cfg.lookupValue("sim.intraAdc", *pintraAdc);
	cfg.lookupValue("sim.myelinAdc", *pmyelinAdc);
	cfg.lookupValue("sim.extraT2", *pextraT2);
	cfg.lookupValue("sim.intraT2", *pintraT2);
	cfg.lookupValue("sim.myelinT2", *pmyelinT2);
	cfg.lookupValue("sim.spaceScale", *pspaceScale);
	cfg.lookupValue("sim.startBox", *pstartBoxSize);
	cfg.lookupValue("sim.permeability", *ppermeability);
	cfg.lookupValue("sim.searchMethod", *psearchMethod);
	const char *tempOutFileName;
	cfg.lookupValue("sim.outFile", tempOutFileName);
	outFileName = new char[strlen(tempOutFileName)];
	strcpy(outFileName, tempOutFileName);

	/////////////////////////////////////////////////////////////////////////
	// Read the gradient information from the gradient file specified in the config file
	/////////////////////////////////////////////////////////////////////////
	const char *gradsFile;
	if (cfg.lookupValue("sim.gradsFile", gradsFile)) {
		printf("Loading gradient sequence from %s.\n", gradsFile);
		FILE *gradsFilePtr = fopen(gradsFile,"r");
		float ld, bd, ro, gx, gy, gz;
		gradStruct *prevGrad = NULL, *curGrad = NULL;
		while (!feof(gradsFilePtr)){
			int nScan = fscanf(gradsFilePtr, "%g %g %g %g %g %g", &ld, &bd, &ro, &gx, &gy, &gz);	// nScan is the number of items successfully read
			if (nScan == 6){									// We have read a line that can specify a gradient
				numGrads++;
				curGrad = new gradStruct;
				curGrad->lDelta = ld;
				curGrad->bDelta = bd;
				curGrad->readOut = ro;
				curGrad->gx = gx;
				curGrad->gy = gy;
				curGrad->gz = gz;
				curGrad->next = NULL;
				if (prevGrad != NULL) prevGrad->next = curGrad;
				if (gGrads==NULL) gGrads = curGrad;
				prevGrad = curGrad;
			}
		}

		fclose(gradsFilePtr);
	}

	//////////////////////////////////////////////////////////////////////////
	// Read the fiber parameters from the cfg file
	//////////////////////////////////////////////////////////////////////////
	const char *fiberFileConst;
	cfg.lookupValue("fibers.fiberFile", fiberFileConst);
	fiberFile = (char *)malloc((strlen(fiberFileConst)+1)*sizeof(char));
	strcpy(fiberFile, fiberFileConst);
	//printf("(In readInputs): Reading fibers from file (%s)\n",fiberFile);


	///////////////////////////////////////////////////////////////////////////////////////
	// See if information in configuration file is overwritten by command line flags.
	///////////////////////////////////////////////////////////////////////////////////////
	
	// See if the command line specifies whether we should display graphics	
	if(cutCheckCmdLineFlag(argc, (const char**) argv, "disp")){
		useDisplay = true;
	}	
	else if(cutCheckCmdLineFlag(argc, (const char**) argv, "nodisp")){
		useDisplay = false;
	}

	if(cutCheckCmdLineFlag(argc, (const char**) argv, "gpu")){
		useGpu = true;
	}	
	else if(cutCheckCmdLineFlag(argc, (const char**) argv, "cpu")){
		useGpu = false;
	}

	if(cutCheckCmdLineFlag(argc, (const char**) argv, "w")){
		writeOut = true;
	}	
	else if(cutCheckCmdLineFlag(argc, (const char**) argv, "nw")){
		writeOut = false;
	}


	float cmdValf = NAN;
	int cmdVali = NAN;
	char *cmdVals;

        if(cutGetCmdLineArgumenti( argc, (const char**) argv, "stepsPerUpdate", &cmdVali)){
		*pstepsPerUpdate = cmdVali;
        }

        if(cutGetCmdLineArgumenti( argc, (const char**) argv, "numSpins", &cmdVali)){
		*pnumSpins = cmdVali;
        }

	if(cutGetCmdLineArgumenti( argc, (const char**) argv, "searchMethod", &cmdVali)){
		*psearchMethod = cmdVali;
        }

        if(cutGetCmdLineArgumentf( argc, (const char**) argv, "perm", &cmdValf)){
		*ppermeability = cmdValf;
        }

	if(cutGetCmdLineArgumentf( argc, (const char**) argv, "startBox", &cmdValf)){
		*pstartBoxSize = cmdValf;
        }

	if(cutGetCmdLineArgumentstr( argc, (const char**) argv, "fiberFile", &cmdVals)){
		fiberFile = (char *)malloc((strlen(cmdVals)+1)*sizeof(char));
		strcpy(fiberFile, cmdVals);
	}
}


/////////////////////////////////////////////////////////////////////////////////////
			// We now calculate the diffusion tensor and its eigenvalues and eigenvectors 
			// The method and the following explanation is from dSimFitTensor.m:
			//	Fit the Stejskal-Tanner equation: S(b) = S(0) exp(-b ADC),
			//	where S(b) is the image acquired at non-zero b-value, and S(0) is
			//	the image acquired at b=0. Thus, we can find ADC with the
			//	following:
			//		ADC = -1/b * log( S(b) / S(0) )
			// 	But, to avoid divide-by-zero, we need to add a small offset to
			//	S(0). We also need to add a small offset to avoid log(0).
			//
			// Note: Using floats in JAMA::SVD causes numerical inaccuracies, so we compute 
			//	everything in double format.
			/////////////////////////////////////////////////////////////////////////////////////
void computeTensor(uint index, double tensors[][3][3], double eigenvecs[][3][3], float lambdas[][3], float mds[], float fas[], float rds[]){
	double* gradDir = new double[3*numGrads];		// gradDir will contain the gradient components, scaled to have norm 1
	double* bVals = new double[numGrads];			// bVals will contain the b-values
	double offset = 0.000001;				// This offset will be used to avoid divide-by-zero errors
	double norm;						// Will contain the norm of each gradient - used for scaling to norm 1
	double logB0 = 0;					// Will contain the average log value of the measured signal for zero gradients
	double* logDw = new double[numGrads];			// Will contain the log values of the measured signal for non-zero gradients
	uint numZeroGrads = 0, numNonZeroGrads = 0;		// Counters for the number of zero and non-zero gradients	
	uint* nonZeroGradsInd = new uint[numGrads];		

	///////////////////////////////////////////////////////////////////
	// Compute gradDir, bVals, logDw and logB0
	///////////////////////////////////////////////////////////////////
	for (uint j=0; j<numGrads; j++){
		norm = sqrt(gradientDataResults[j*numOutputs+3]*gradientDataResults[j*numOutputs+3]+gradientDataResults[j*numOutputs+4]*gradientDataResults[j*numOutputs+4]+gradientDataResults[j*numOutputs+5]*gradientDataResults[j*numOutputs+5]);
		printf("Signal for gradient %u and compartment %i: %g\n", j+1, index-1, gradientDataResults[j*numOutputs+6+index]);
		if (norm==0){
			gradDir[j*3+0] = 0;
			gradDir[j*3+1] = 0;
			gradDir[j*3+2] = 0;
			logB0 += log(gradientDataResults[j*numOutputs+6+index]+offset);
			numZeroGrads++;
		} else {
			gradDir[j*3+0] = (double) gradientDataResults[j*numOutputs+3]/norm;
			gradDir[j*3+1] = (double) gradientDataResults[j*numOutputs+4]/norm;
			gradDir[j*3+2] = (double) gradientDataResults[j*numOutputs+5]/norm;
			logDw[numNonZeroGrads] = log(gradientDataResults[j*numOutputs+6+index]+offset);
			nonZeroGradsInd[numNonZeroGrads] = j;
			numNonZeroGrads++;
		}

		bVals[j] = pow((2.0*3.1415926535*42576.0)*norm/1000000000*gradientDataResults[j*numOutputs+0],2)*(gradientDataResults[j*numOutputs+1]-gradientDataResults[j*numOutputs+0]/3.0);
	}

	logB0 = logB0/numZeroGrads;

	double* bv = new double[numNonZeroGrads*3];			// bv will contain the normalized components of all non-zero gradients
	double errorMargin = 0.000001;					// We will round off matrices to precision errormargin, to avoid small-value numerical errors
	TNT::Array2D<double> adc(numNonZeroGrads,1);			// ADC for each non-zero gradient according to ADC = -1/b * log( S(b) / S(0) )			
	TNT::Array2D<double> m(numNonZeroGrads,6);			// Rearrangement of normalized non-zero gradient components


	for (uint k=0; k<numNonZeroGrads; k++){
		adc[k][0] = -1/bVals[nonZeroGradsInd[k]]*(logDw[k]-logB0);

		bv[k*3+0] = gradDir[nonZeroGradsInd[k]*3+0];
		bv[k*3+1] = gradDir[nonZeroGradsInd[k]*3+1];
		bv[k*3+2] = gradDir[nonZeroGradsInd[k]*3+2];

		m[k][0] = bv[k*3+0]*bv[k*3+0];
		m[k][1] = bv[k*3+1]*bv[k*3+1];
		m[k][2] = bv[k*3+2]*bv[k*3+2];
		m[k][3] = 2*bv[k*3+0]*bv[k*3+1];
		m[k][4] = 2*bv[k*3+0]*bv[k*3+2];
		m[k][5] = 2*bv[k*3+1]*bv[k*3+2];
	}

	// Round off m to avoid small-value numerical errors
	for (int n=0; n<m.dim1(); n++){
		for(int m1=0; m1<m.dim2(); m1++){
			m[n][m1] = floor(m[n][m1]*1000000.0f + 0.5)/1000000.0f;
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////
	// We now find the pseudo-inverse of m by doing an SVD decomposition
	///////////////////////////////////////////////////////////////////////////////////////
	JAMA::SVD<double> svd(m);
	TNT::Array2D<double> S;
	TNT::Array2D<double> U;
	TNT::Array2D<double> V;

	svd.getS(S);
	svd.getU(U);
	svd.getV(V);

	// Round off S, U and V to avoid small-value numerical errors
	for (int p=0; p<S.dim1(); p++){
		for(int q=0; q<S.dim2(); q++){
			S[p][q] = floor(S[p][q]*1000000.0f + 0.5)/1000000.0f;
		}
	}

	for (int p=0; p<U.dim1(); p++){
		for(int q=0; q<U.dim2(); q++){
			U[p][q] = floor(U[p][q]*1000000.0f + 0.5)/1000000.0f;
		}
	}

	for (int p=0; p<V.dim1(); p++){
		for(int q=0; q<V.dim2(); q++){
			V[p][q] = floor(V[p][q]*1000000.0f + 0.5)/1000000.0f;
		}
	}

	// Find the inverse matrix of S
	TNT::Array2D<double> S_inv(S.dim2(),S.dim1());
	for (int n=0; n<S.dim1(); n++){
		for (int m=0; m<S.dim2(); m++){
			if (S[n][m] == 0){
				S_inv[m][n] = 0;
			} else {
				S_inv[m][n] = 1/S[n][m];
			}
		}
	}

	// Transpose U and V (they are real, so we don't need to take complex conjugates)
	TNT::Array2D<double> Ustar(U.dim2(),U.dim1());
	for (int n=0; n<U.dim1(); n++){
		for (int m=0; m<U.dim2(); m++){
			Ustar[m][n] = U[n][m];
		}
	}

	TNT::Array2D<double> Vstar(V.dim2(),V.dim1());
	for (int n=0; n<V.dim1(); n++){
		for (int m=0; m<V.dim2(); m++){
			Vstar[m][n] = V[n][m];
		}
	}

	// Now calculate the pseudo-inverse of m by m_inv = V*S_inv*Ustar
	TNT::Array2D<double> m_inv(m.dim2(),m.dim1());
	TNT::Array2D<double> S_invUstar(S_inv.dim1(),U.dim1());

	S_invUstar = matmult(S_inv,Ustar);
	m_inv = matmult(V,S_invUstar);

	// Round off m_inv to avoid small-value numerical errors
	for (int p=0; p<m_inv.dim1(); p++){
		for(int q=0; q<m_inv.dim2(); q++){
			m_inv[p][q] = floor(m_inv[p][q]*1000000.0f + 0.5)/1000000.0f;
		}
	}

	// Use the pseudo-inverse to compute the diffusion tensor D
	TNT::Array2D<double> coef(m_inv.dim1(),1);
	TNT::Array2D<double> D(3,3);

	coef = matmult(m_inv,adc);
	D[0][0] = coef[0][0]; D[0][1] = coef[3][0]; D[0][2] = coef[4][0];
	D[1][0] = coef[3][0]; D[1][1] = coef[1][0]; D[1][2] = coef[5][0];
	D[2][0] = coef[4][0]; D[2][1] = coef[5][0]; D[2][2] = coef[2][0];

	// Get the eigenvalues and eigenvectors of the diffusion tensor D
	JAMA::Eigenvalue<double> eig(D);
	TNT::Array1D<double> eigenvalues;
	TNT::Array2D<double> eigenvectors;
	eig.getRealEigenvalues(eigenvalues);
	eig.getV(eigenvectors);

	// l1 to l3 are the largest to the smallest eigenvalues of the diffusion tensor
	float l1 = (float) eigenvalues[2];
	float l2 = (float) eigenvalues[1];
	float l3 = (float) eigenvalues[0];

	// Compute the mean diffusivity, fractional anisotropy and radial diffusivity
	float md = (l1+l2+l3)/3.0f;
	float fa = sqrt(3.0/2.0)*sqrt(pow(l1-md,2)+pow(l2-md,2)+pow(l3-md,2))/sqrt(pow(l1,2)+pow(l2,2)+pow(l3,2));
	float rd = (l2+l3)/2.0f;

	printf("*\n");
	// Compute parameters for plotting ellipsoid if we are looking at total signal
	if (index==0){
		// [ex,ey,ez] is the eigenvector corresponding to the largest eigenvalue (should be pointed in the main diffusion direction)
		ex = (float) eigenvectors[0][2];
		ey = (float) eigenvectors[1][2];
		ez = (float) eigenvectors[2][2];

		printf("Main eigenvector: [%g, %g, %g]\n", ex,ey,ez);


		// Theta is the angle between [ex,ey,ez] and the x-axis
		theta = acos(ex/sqrt(ex*ex+ey*ey+ez*ez))*180/3.1415926535;

		printf("Theta: %g\n", theta);

		// We will plot the ellipsoid with axes ell1, ell2 and ell3, scales by ellipse_scalefactor
		float ellipse_scalefactor = 0.7;
		ell1 = ellipse_scalefactor*1;
		ell2 = ellipse_scalefactor*sqrt(l2/l1);
		ell3 = ellipse_scalefactor*sqrt(l3/l1);

		printf("ell1, ell2, ell3: %g, %g, %g\n", ell1, ell2, ell3);
	}

	// Print out the results
	printf("Eigenvector matrix : \n");
	for (int n=0; n<eigenvectors.dim1(); n++){
		for(int m=0; m<eigenvectors.dim2(); m++){
			printf("%g ",eigenvectors[n][m]);
			eigenvecs[index][n][m] = eigenvectors[n][m];
		}
		printf("\n");
	}

	printf("D = \n");
	for (int n=0; n<3; n++){
		for(int m=0; m<3; m++){
			printf("%g ",D[n][m]);
			tensors[index][n][m] = D[n][m];
		}
		printf("\n");
	}

	printf("l1: %g\n", l1);
	printf("l2: %g\n", l2);
	printf("l3: %g\n", l3);
	printf("md (mean diffusivity): %g\n", md);
	printf("fa (fractional anisotropy): %g\n", fa);
	printf("rd (radial diffusivity): %g\n", rd);
	printf("*\n");
	
	lambdas[index][0] = (float) eigenvalues[0];
	lambdas[index][1] = (float) eigenvalues[1];
	lambdas[index][2] = (float) eigenvalues[2];

	mds[index] = (lambdas[index][0]+lambdas[index][1]+lambdas[index][2])/3.0f;
	fas[index] = sqrt(3.0/2.0)*sqrt(pow(lambdas[index][0]-mds[index],2)+pow(lambdas[index][1]-mds[index],2)+pow(lambdas[index][2]-mds[index],2))/sqrt(pow(lambdas[index][0],2)+pow(lambdas[index][1],2)+pow(lambdas[index][2],2));
	rds[index] = (lambdas[index][0]+lambdas[index][1])/2.0f;

	delete [] gradDir;
	delete [] bVals;
	delete [] logDw;
}


////////////////////////////////////////////////////////////////////////////////////////
// Function name: update.
// Description: Increments the number of timesteps, updates the gradient sequence data
// 		accordingly and then updates the spin data by calling updateSpins (at the
//		bottom of the function). When the gradient sequence is completed, the 
//		function computes the diffusion ellipsoid values and sets the 
//		computedOutput flag to 1.
////////////////////////////////////////////////////////////////////////////////////////
void update(){
	//////////////////////////////////////////////////////////////////
	// We make the following gradient sequence variables static, to
	// make them retain their values over multiple function calls.
	//////////////////////////////////////////////////////////////////
	static int currentRun = 0;				// The number of the current gradient in the sequence (i.e. the run number)
	static gradStruct *curGrad = gGrads;			// Points to the first gradient in the gradient sequence
	static float3 curG;					// Will contain the xyz-components of the current gradient (pointed to by curGrad)
	float3 nullG = {0.0f, 0.0f, 0.0f};			// A zero gradient, will be used to simulate time periods of no gradients
	static int curBigDelta;					// The number of updates in the big delta of the gradient
	static int curLittleDelta;				// The number of updates in the little delta of the gradient
	static int curReadout;					// The number of updates in the read out time of the gradient
	static int nUpdates = -1;				// The number of updates during the current gradient run
	static int curNumUpdates = 0;				// Number of time steps in the current run (=curBigDelta+curLittleDelta+curReadout)
	static bool done;					// 1 if all gradients have finished, 0 otherwise
	double mrSig;
	double mrSigComp[psystem->getNumCompartments()+1];
	

	/////////////////////////////////////////////////////////////////
	// Each update call increments the number of updates by one and
	// the number of time steps by stepsPerUpdate.
	/////////////////////////////////////////////////////////////////
	updateCount++;
	stepCount+=stepsPerUpdate;

	/////////////////////////////////////////////////////////////////
	// Update the gradient information depending on where we are in
	// the gradient sequence.
	/////////////////////////////////////////////////////////////////
	if (gGrads!=NULL){					// A NULL gGrads indicates no DW gradients (an empty gradient sequence)
		if (curNumUpdates <= nUpdates){			// Current DW gradient run not finished, so we continue processing this DW run

			////////////////////////////////////////////////////
			// Print time and signal information if simulation
			// is not finished.
			////////////////////////////////////////////////////
			if (!computedOutput){
				printf("Step Count: %u, Time step: %g, ", stepCount, timeStep);
				printf("MR Signal: %0.4f, ms: %4.2f \n", psystem->getMrSignal(), (float)stepCount*timeStep);
			}

			if (curNumUpdates == 0){							// Reset and then turn on DW gradients
				psystem->resetSpins();
				psystem->setGradient(curG);
				printf("Turning on gradients!\n");
			} else if (curNumUpdates==curLittleDelta && curNumUpdates<curBigDelta){		// Turn off DW grads
				psystem->setGradient(nullG);
				printf("Turning off gradient!\n");
			} else if (curNumUpdates==curBigDelta){						// Apply opposite gradient
				psystem->setGradient(curG*-1.0f);
				printf("Applying opposite gradient!\n");
			} else if (curNumUpdates==curBigDelta+curLittleDelta){				// Turn off DW grads
				psystem->setGradient(nullG);
				printf("Turning off gradient!\n");
			} else if (curNumUpdates==curBigDelta+curLittleDelta+curReadout){		// Measure MR signal for each compartment (and total signal) and save it
				//mrSig = psystem->getMrSignal();
				psystem->getMrSignal(mrSigComp);
				//gradientDataResults[(currentRun-1)*7+6] = mrSig;
				for (uint c=0; c<psystem->getNumCompartments()+1; c++){
					gradientDataResults[(currentRun-1)*numOutputs+6+c] = mrSigComp[c];
				}
				//gradientDataResults[currentRun*numOutputs-1] = mrSig;
			}
			curNumUpdates++;

		} else{							// curNumUpdates > nUpdates, so nothing more to do for current DW run. Load the params for the next DW run
			curNumUpdates = 0;				// Reset the curNumUpdates counter
			if (curGrad == NULL){				// We are at the end of the DW gradient sequence
				done = true;
			} else{
				/////////////////////////////////////////////////////
				// Some gradients remain, update curG, curLittleDelta,
				// curBigDelta and curReadOut accordingly.
				/////////////////////////////////////////////////////
				currentRun++;
				printf("Run no. %i\n", currentRun);
				printf("Running: d=%gms, D=%gms, readout=%gms, G=[%g,%g,%g]mT/m.\n", 
					curGrad->lDelta, curGrad->bDelta, curGrad->readOut, curGrad->gx, curGrad->gy, curGrad->gz);
				curG.x = curGrad->gx;
				curG.y = curGrad->gy;
				curG.z = curGrad->gz;
				printf("Steps per update: %u\n", stepsPerUpdate);
				curLittleDelta = (int)(curGrad->lDelta/(timeStep*(float)stepsPerUpdate)+0.5);
				curBigDelta = (int)(curGrad->bDelta/(timeStep*(float)stepsPerUpdate)+0.5);
				curReadout = (int)(curGrad->readOut/(timeStep*(float)stepsPerUpdate)+0.5);
				nUpdates = curBigDelta+curLittleDelta+curReadout;

				//////////////////////////////////////////////////////
				// Store the gradient parameters in the gradientDataResults,
				// which will be used for later computation and writing
				// to an output file. The measured signal for each
				// gradient will also be stored in gradientDataResults.
				//////////////////////////////////////////////////////
				gradientDataResults[(currentRun-1)*numOutputs+0] = curGrad->lDelta;
				gradientDataResults[(currentRun-1)*numOutputs+1] = curGrad->bDelta;
				gradientDataResults[(currentRun-1)*numOutputs+2] = curGrad->readOut;
				gradientDataResults[(currentRun-1)*numOutputs+3] = curG.x;
				gradientDataResults[(currentRun-1)*numOutputs+4] = curG.y;
				gradientDataResults[(currentRun-1)*numOutputs+5] = curG.z;

				curGrad = curGrad->next;				// Update curGrad to point to the next gradient in the sequence
			}
		}


		///////////////////////////////////////////////////////////////////
		// If gradient sequence is finished, we write the results into an
		// output file and compute the diffusion ellipsoid parameters
		// (i.e. eigenvectors and eigenvalues of the diffusion tensor)
		///////////////////////////////////////////////////////////////////
		if (done & !computedOutput){
			printf("Stepcount: %u \n", stepCount);
			printf("All runs finished!\n");
			
			double tensors[psystem->getNumCompartments()+1][3][3]; 
			double eigenvecs[psystem->getNumCompartments()+1][3][3];
			float lambdas[psystem->getNumCompartments()+1][3]; 
			float mds[psystem->getNumCompartments()+1];
			float fas[psystem->getNumCompartments()+1];
			float rds[psystem->getNumCompartments()+1];

			printf("*\nEigenvectors, diffusion tensor, eigenvalues, md, fa, rd for total signal\n");
			computeTensor(0,tensors,eigenvecs,lambdas,mds,fas,rds);
			for (uint c=0; c<psystem->getNumCompartments(); c++){
				printf("*\nEigenvectors, diffusion tensor, eigenvalues, md, fa, rd for compartment %u\n", c);
				computeTensor(c+1,tensors,eigenvecs,lambdas,mds,fas,rds);
			}

			for(uint i=0; i<psystem->getNumCompartments()+1; i++){
				printf("Tensor for compartment %i: %g, %g, %g, %g, %g, %g, %g, %g, %g\n", i-1, tensors[i][0][0], tensors[i][0][1], tensors[i][0][2], tensors[i][1][0], tensors[i][1][1], tensors[i][1][2], tensors[i][2][0], tensors[i][2][1], tensors[i][2][2]);
				printf("Eigenvector for compartment %i: [%g,%g,%g]\n", i-1, eigenvecs[i][0][2], eigenvecs[i][1][2], eigenvecs[i][2][2]);
				printf("Eigenvalues for compartment %i: %g, %g, %g\n", i-1, lambdas[i][0], lambdas[i][1], lambdas[i][2]);
				printf("Mean diffusivity for compartment %i: %g\n", i-1, mds[i]);
				printf("Fractional anisotropy for compartment %i: %g\n", i-1, fas[i]);
				printf("Radial diffusivity for compartment %i: %g\n", i-1, rds[i]);
			}

			computedOutput = 1;

			///////////////////////////////////////////////////////
			// Write the results to an output file, if specified, 
			// and also to the terminal.
			///////////////////////////////////////////////////////
			outFilePtr = fopen(outFileName,"at");
			if(outFilePtr!=NULL & writeOut){
				time_t now = time(NULL);
				fprintf(outFilePtr,"\n%% * * * %s", ctime(&now));
				fprintf(outFilePtr, "if (exist('m','var')), m=m+1; else, m=1; end\n");
				fprintf(outFilePtr, "s=0;\n");
				fprintf(outFilePtr, "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n%% Input parameters \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
				fprintf(outFilePtr, "numSpins(m) = %u;\n", numSpins);
				fprintf(outFilePtr, "useGpu(m) = %d;\n", useGpu);
				fprintf(outFilePtr, "fibersFile{m} = '%s';\n", fiberFile);
				fprintf(outFilePtr, "useDisplay(m) = %d;\n", useDisplay);
				fprintf(outFilePtr, "stepsPerUpdate(m) = %i;\n", stepsPerUpdate);
				fprintf(outFilePtr, "gyroMagneticRatio(m) = %g;\n", gyroMagneticRatio);
				fprintf(outFilePtr, "timeStep(m) = %g;\n", timeStep);
				fprintf(outFilePtr, "extraAdc(m) = %g;\n", extraAdc);
				fprintf(outFilePtr, "intraAdc(m) = %g;\n", intraAdc);
				fprintf(outFilePtr, "myelinAdc(m) = %g;\n", myelinAdc);
				fprintf(outFilePtr, "spaceScale(m) = %g;\n", spaceScale);
				fprintf(outFilePtr, "permeability(m) = %g;\n", permeability);
				fprintf(outFilePtr, "\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n%% Simulation data \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n");

				fprintf(outFilePtr, "%%The following variables give the diffusion tensor, eigenvector matrix, eigenvalues\n");
				fprintf(outFilePtr, "%%(eigenvalue no. i corresponds to column no. i in the eigenvector matrix), mean diffusivity,\n");
				fprintf(outFilePtr, "%%fractional anisotropy and radial diffusivity for the whole volume (index 1) and each\n");
				fprintf(outFilePtr, "%%compartment (index 2 for compartment 0, index 3 for compartment 1, etc).\n");
				for (uint c=0; c<psystem->getNumCompartments(); c++){
					fprintf(outFilePtr, "tensor{m}{%u} = [%g, %g, %g; %g, %g, %g; %g, %g, %g];\n", c+1, tensors[c][0][0], tensors[c][0][1], tensors[c][0][2], tensors[c][1][0], tensors[c][1][1], tensors[c][1][2], tensors[c][2][0], tensors[c][2][1], tensors[c][2][2]);
					fprintf(outFilePtr, "eigenvecs{m}{%u} = [%g, %g, %g; %g, %g, %g; %g, %g, %g];\n", c+1, eigenvecs[c][0][0], eigenvecs[c][0][1], eigenvecs[c][0][2], eigenvecs[c][1][0], eigenvecs[c][1][1], eigenvecs[c][1][2], eigenvecs[c][2][0], eigenvecs[c][2][1], eigenvecs[c][2][2]);
					fprintf(outFilePtr, "eigenvals{m}{%u} = [%g,%g,%g];\n", c+1, lambdas[c][0], lambdas[c][1], lambdas[c][2]);
					fprintf(outFilePtr, "md{m}(%u) = %g;\n", c+1, mds[c]);
					fprintf(outFilePtr, "fa{m}(%u) = %g;\n", c+1, fas[c]);
					fprintf(outFilePtr, "rd{m}(%u) = %g;\n", c+1, rds[c]);
				}

				fprintf(outFilePtr, "%%The following variables give results for each gradient run.\n");
				fprintf(outFilePtr, "%%mrSigTotal{m}(c,s) is the total signal after gradient run s during diffusion simulation no. m\n");
				fprintf(outFilePtr, "%%mrSigCompartments{m}(c,s) is the contribution of compartment no. (c-2) to the \n");
				fprintf(outFilePtr, "%%total signal after gradient run s during diffusion simulation no. m. c=1 represents all compartments. \n");

				for (uint i=0; i<numGrads; i++){
					printf("Gradient run no. %u\n", i);
					printf("Little delta: %g\n", gradientDataResults[i*numOutputs+0]);
					printf("Big delta: %g\n", gradientDataResults[i*numOutputs+1]);
					printf("Readout time: %g\n", gradientDataResults[i*numOutputs+2]);
					printf("Gradient: [%g,%g,%g]\n", gradientDataResults[i*numOutputs+3],gradientDataResults[i*numOutputs+4],gradientDataResults[i*numOutputs+5]);
					printf("Total MR signal: %g\n", gradientDataResults[i*numOutputs+6]);
					for (uint c=0; c<psystem->getNumCompartments(); c++){
						printf("Signal contribution from compartment %u: %g\n", c, gradientDataResults[i*numOutputs+6+c+1]);
					}

					fprintf(outFilePtr, "\ns=s+1;\n");
					fprintf(outFilePtr, "delta{m}(s) = %g;\n", gradientDataResults[i*numOutputs+0]);
					fprintf(outFilePtr, "Delta{m}(s) = %g;\n", gradientDataResults[i*numOutputs+1]);
					fprintf(outFilePtr, "readOut{m}(s) = %g;\n", gradientDataResults[i*numOutputs+2]);
					fprintf(outFilePtr, "dwGrads{m}(:,s) = [%g,%g,%g];\n\n", gradientDataResults[i*numOutputs+3], gradientDataResults[i*numOutputs+4], gradientDataResults[i*numOutputs+5]);
					fprintf(outFilePtr, "mrSigTotal{m}(s) = %g;\n\n", gradientDataResults[i*numOutputs+6]);
					for (uint c=0; c<psystem->getNumCompartments(); c++){
						fprintf(outFilePtr, "mrSigCompartments{m}(%u,s) = %g;\n", c+1, gradientDataResults[i*numOutputs+6+c+1]);
					}
		         		fflush(outFilePtr);
				}
			}
			fclose(outFilePtr);
			
		}
	}

	// We move the spins over the update period and calculate the resulting MR signal
	//printf("Test\n");
	psystem->updateSpins(timeStep,stepsPerUpdate);
	//printf("Test2\n");
}


/////////////////////////////////////////////////////////////////////////////////////////////
// Function name: 	initGL
// Description:		Initialize OpenGL parameters and check whether system is OpenGL 
//			compatible
/////////////////////////////////////////////////////////////////////////////////////////////
void initGL(){
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object")) {
		printf("Required OpenGL extensions missing.");
		exit(-1);
	}
	glEnable(GL_DEPTH_TEST);
	glClearColor(0.5, 0.5, 0.5, 1.0);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE);
	glutReportErrors();
}


///////////////////////////////////////////////////////////////////////////////////////////////
// Function name:	display
// Description:		Displays the simulated volume with fibers. Also displays spins and runs
//			the update function to recompute spin movement and signal if the gradient
//			sequence is not finished.
///////////////////////////////////////////////////////////////////////////////////////////////
void display(){
	
	////////////////////////////////////////////////////////////////
	// Update spin information if gradient sequence is not finished
	////////////////////////////////////////////////////////////////
	if (!computedOutput){
		//update();
    		if(!bPause){
    	
        		update();
        		renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumSpins());
        		psystem->setColorFromSignal();
   		}else{
         		usleep(15000);
    		}
		//printf("Done running update\n");
	}
	//printf("Test3");

	////////////////////////////////////////////////////////////////
	// Set parameters for rendering and movement
	////////////////////////////////////////////////////////////////
	//renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumSpins());
	//psystem->setColorFromSpin();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	for (int c = 0; c < 3; ++c){
		camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
		camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
	}
	glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
	glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
	glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

	glGetFloatv(GL_MODELVIEW_MATRIX, modelview);

	/////////////////////////////////////////////////////////////////
	// Draw the rectangle defining our simulation volume
	/////////////////////////////////////////////////////////////////
	glColor3f(1.0, 1.0, 1.0);
	glutWireCube(2.0);

	if (computedOutput){
		////////////////////////////////////////////////////////
		// If gradient runs are finished and ellipsoid parameters
		// have been computed, we draw the resulting ellipsoid by
		// drawing a sphere and stretching and rotating it.
		////////////////////////////////////////////////////////
		GLUquadricObj* obj = gluNewQuadric();
		glRotatef(theta,0,-ez,ey);
		glScalef(ell1,ell2,ell3);
		glColor4f(0.4, 0.4, 0.0, 0.1);
		gluSphere(obj,1.0,20.0,20.0);
		glScalef(1/ell1,1/ell2,1/ell3);
		glRotatef(-theta,0,-ez,ey);
		gluDeleteQuadric(obj);
	} else {
		/////////////////////////////////////////////////////////
		// If gradient runs are not finished, we render the spins.
		/////////////////////////////////////////////////////////
		renderer->display(displayMode);
	}

	////////////////////////////////////////////////////////////////////////
	// Render triangles
	////////////////////////////////////////////////////////////////////////
	uint nFibers = psystem->getNumFibers();
	float3 p1, p2, p3;
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);					// Simple transparency

	//printf("Triangles in fiber %u in membrane type %u: %u\n", 8, 2, psystem->getNumTriInFiberMembrane(2,8));
	////std::cin.get();

	uint renderedMembranes[2] = {1,2};
	for (uint m = 0; m < 2; m++){
		uint membraneIndex = renderedMembranes[m];
		for (uint fiberIndex = 0; fiberIndex<nFibers; fiberIndex++){					// Loop through all fibers
			uint numMyelinTri = psystem->getNumTriInFiberMembrane(membraneIndex, fiberIndex);	// Get number of triangles of membrane type membraneIndex in fiber fiberIndex
			for (uint triIndex = 0; triIndex<numMyelinTri; triIndex++){				// Loop through all the triangles in the myelin sheath
				uint nTri = psystem->getTriInFiberArray(membraneIndex, fiberIndex, triIndex);
				p1 = psystem->getTriPoint(nTri,0); p2 = psystem->getTriPoint(nTri,1); p3 = psystem->getTriPoint(nTri,2);
				glPushMatrix();
				if (membraneIndex == 1){
					glColor4f(0.0, 0.3, 0.1, myelinAlpha);
				} else {
					glColor4f(0.0, 0.0, 0.8, gliaAlpha);
				}
				glBegin(GL_TRIANGLES);
				glVertex3f(p1.x,p1.y,p1.z);
				glVertex3f(p2.x,p2.y,p2.z);
				glVertex3f(p3.x,p3.y,p3.z);
				glEnd();
				glPopMatrix();
			}
		}
	}
	glDisable(GL_BLEND);
	
	glutSwapBuffers();
	glutReportErrors();
}


///////////////////////////////////////////////////////////////////////
// Function name:	reshape
// Description:		Function for enable changing of the window size.
///////////////////////////////////////////////////////////////////////
void reshape(int w, int h){
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (float) w / (float) h, 0.1, 10.0);

	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, w, h);

	renderer->setWindowSize(w,h);
	renderer->setFOV(60.0);
}


///////////////////////////////////////////////////////////////////////
// Function name:	mouse
// Description:		Enables control of the window via the mouse.
///////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y){
	int mods;
	
	if (state == GLUT_DOWN)
		buttonState |= 1<<button;
	else if (state == GLUT_UP)
		buttonState == 0;

	mods = glutGetModifiers();
	if (mods & GLUT_ACTIVE_SHIFT){
		buttonState = 2;
	} else if (mods & GLUT_ACTIVE_CTRL){
		buttonState = 3;
	}

	ox = x;
	oy = y;
	glutPostRedisplay();
}


//////////////////////////////////////////////////////////////////////
// Function name:	motion
// Description:		Controls camera motion.
//////////////////////////////////////////////////////////////////////
void motion(int x, int y){
	float dx, dy;
	dx = x - ox;
	dy = y - oy;

	if (buttonState == 3){
		// left+middle = zoom
		camera_trans[2] += (dy / 100.0) * 0.5 * fabs(camera_trans[2]);
	} else if (buttonState & 2){
		// middle = translate
		camera_trans[0] += dx / 100.0;
		camera_trans[1] -= dy / 100.0;
	} else if (buttonState & 1){
		// left = rotate
		camera_rot[0] += dy / 5.0;
		camera_rot[1] += dx / 5.0;
	}

	ox = x;
	oy = y;
	glutPostRedisplay();
}


/////////////////////////////////////////////////////////////////////
// Function name:	idle
// Description:		Sets the callback function when user is idle.
/////////////////////////////////////////////////////////////////////
void idle(void){
	glutPostRedisplay();
}


void key(unsigned char key, int /*x*/, int /*y*/)
{

    switch (key) 
    {
    case ' ':
        bPause = !bPause;
        break;
   /* case 13:
        if(bPause){
           time_in_sec = psystem->update(timestep, iterations); 
           renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumSpins());
           stepCount++;
        }
        break;
    case '\033':
    case 'q':
        exit(0);
        break;
    case 'd':
        psystem->dumpGrid();
        break;
    case 'u':
        psystem->dumpSpins(0, 1);
        break;

    case 'r':
        renderSpins = !renderSpins;
        break;

    case '1':
        psystem->reset(SpinSystem::CONFIG_GRID);
        stepCount = 0;
        break;
    case '2':
        psystem->reset(SpinSystem::CONFIG_RANDOM);
        stepCount = 0;
        break;
    case '5':
        {
            // color a sphere in the center
            float pos[3] = { 0.0f, 0.0f, 0.0f };
            float color[4] = { 1.0f, 0.05f, 0.1f, 1.0f };
            psystem->colorSphere(pos, 0.2, color); 
        }
        break;

    case 'w':
        wireframe = !wireframe;
        break;

    case 't':
        renderFibers = !renderFibers;
        break;

    case 's':
        renderSpheres = !renderSpheres;
        break;

    case 'T':
        if(fiberAlpha==1.0) fiberAlpha = 0.3;
        else fiberAlpha = 1.0;
        break;

    case 'h':
        displaySliders = !displaySliders;
        break;
    case 'c':
	renderCubes = !renderCubes;	
	break;*/
    }

    glutPostRedisplay();
}


/////////////////////////////////////////////////////////////////////
// Function name:	mainMenu
// Description:		Sets the callback function for creating a user
//			menu - entries will be added to the menu in
//			initMenus.
/////////////////////////////////////////////////////////////////////
void mainMenu(int i){
	key((unsigned char) i, 0, 0);
}


/////////////////////////////////////////////////////////////////////
// Function name:	initMenus
// Description:		Create the main menu and add entries to the menu.
//			Currently not in use.
/////////////////////////////////////////////////////////////////////
void initMenus(){
	glutCreateMenu(mainMenu);
	glutAddMenuEntry("Reset block [1]", '1');
	glutAddMenuEntry("Reset random [2]", '2');
	glutAddMenuEntry("Toggle animation [ ]", ' ');
	glutAddMenuEntry("Step animation [ret]", 13);
	glutAddMenuEntry("Toggle fibers [t]", 't');
	glutAddMenuEntry("Toggle spheres [s]", 's');  
	glutAddMenuEntry("Toggle sliders [h]", 'h');
	glutAddMenuEntry("Toggle grid cubes [c]", 'c');
	glutAddMenuEntry("Quit (esc)", '\033');
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}



//////////////////////////////////////////////////////////////////////
// Program  main
//////////////////////////////////////////////////////////////////////
using namespace libconfig;
using namespace std;

int main( int argc, char** argv){

	printf("Start diffusion run\n");

	FILE *fiberFilePtr;

	///////////////////////////////////////////////////////////////////////////////////////
	// The name of the configuration file is assumed to be sim.cfg unless another name is 
	// detected using the function cutGetCmdLineArgumentstr.
	// To use this function we need to use the flag -lcutil in our compile statement
	// See description at stingnet.com
	///////////////////////////////////////////////////////////////////////////////////////
	char *configFilename;
	if (!cutGetCmdLineArgumentstr(argc, (const char**) argv, "config", &configFilename)){
		configFilename = "sim.cfg";
	}

	///////////////////////////////////////////////////////////////////////////////////////
	// Read the simulation parameters from the configuration file. Overwrite the default 
	// program values.
	///////////////////////////////////////////////////////////////////////////////////////
	printf("useDisplay: %i\n", useDisplay);
	printf("numSpins: %u\n", numSpins);
	printf("stepsPerUpdate: %i\n", stepsPerUpdate);
	printf("triSearchMethod: %u\n", triSearchMethod);
	printf("(In main): Reading fibers from file (%s)\n",fiberFile);
	readInputs(argc, argv, configFilename, &stepsPerUpdate, &useGpu, &useDisplay, &writeOut, &triSearchMethod, &numSpins, &gyroMagneticRatio, 
		&timeStep, &extraAdc, &intraAdc, &myelinAdc, &extraT2, &intraT2, &myelinT2, &spaceScale, &permeability, &startBoxSize, outFileName);

	printf("triSearchMethod: %u\n", triSearchMethod);
	//std::cin.get();

	printf("useDisplay: %i\n", useDisplay);
	printf("numSpins: %u\n", numSpins);
	printf("stepsPerUpdate: %i\n", stepsPerUpdate);
	printf("(In main): Reading fibers from file (%s)\n",fiberFile);
	//std::cin.get();

	///////////////////////////////////////////////////////////////////////////////////////
	// Initialize graphics if output is rendered to screen.
	///////////////////////////////////////////////////////////////////////////////////////
	if (useDisplay){
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
		glutInitWindowSize(640, 480);
		glutCreateWindow("Diffusion Simulator");
		initGL();
	}

	// Create a new spin system
	psystem = new SpinSystem(numSpins, useGpu, spaceScale, gyroMagneticRatio, useDisplay, triSearchMethod, reflectionType, extraAdc, myelinAdc, intraAdc, permeability, 
				extraT2, myelinT2, intraT2, timeStep, startBoxSize);
	// Define fibers for the spin system
	psystem->initFibers(fiberFile);
	// Assign triangles to each cube
	if (triSearchMethod == 0){
		printf("Populating cubes\n");
		psystem->populateCubes();
		printf("Done populating cubes\n");
		//std::cin.get();
	}
	psystem->constructAllRTrees();
	printf("RTrees constructed\n");
	////std::cin.get();
	// Allocate memory for the spin system and abort if there is an error.
	printf("Allocating memory\n");
	if(!psystem->build()) exit(-1);
	printf("Done allocating memory\n");
	// Randomize the spins
	printf("Resetting spins\n");
	psystem->resetSpins();
	printf("Done resetting spins\n");

	///////////////////////////////////////////////////////////////////////////////////////
	// Create an array for storing gradient information and results.
	///////////////////////////////////////////////////////////////////////////////////////
	numOutputs = 6 + psystem->getNumCompartments() + 1;
	//printf("numOutputs: %u\n", numOutputs);
	////std::cin.get();
	gradientDataResults = new float[numGrads*numOutputs];
	memset(gradientDataResults, 0, numGrads*numOutputs*sizeof(float));


	// Do an update to initialize the GPU stuff
	printf("Doing one step update\n");
	psystem->updateSpins(timeStep,1);
	printf("Finished with one step update\n");

	if (useDisplay){
		////////////////////////////////////////////////////////////////
		// We render the simulation on the computer screen. The computation
		// takes place inside the display function.
		////////////////////////////////////////////////////////////////
		renderer = new SpinRenderer;
		renderer->setSpinRadius(0.005);
		renderer->setColorBuffer(psystem->getColorBuffer());
		initMenus();
		glutDisplayFunc(display);
		glutReshapeFunc(reshape);
		glutMouseFunc(mouse);
		glutMotionFunc(motion);
		glutIdleFunc(idle);
		glutMainLoop();
	} else {
		///////////////////////////////////////////////////////////////
		// Nothing is rendered to the computer screen, we run the computation
		// until the gradient runs are finished and then exit.
		///////////////////////////////////////////////////////////////
		while(!computedOutput){    
			update();
		}
		//exit(0);
	}

	if (psystem) delete psystem;
	return 0;
}
