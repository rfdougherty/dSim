/*_______________________________________________________________________________
 * MR Diffusion Simulation (dSim)
 *_______________________________________________________________________________
 *
 * Copyright 2008 Bob Dougherty (bobd@stanford.edu)
 *
 * Portions of this code are based on the NVidia 'Spins' demo 
 * distributed with the CUDA SDK and are thus copyright 1993-2007 
 * NVIDIA Corporation. Details are provided in the individual source 
 * files (e.g., renderSpins.cpp).
 */

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <math.h>
#include <cutil.h>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut.h>
#include <GL/gl.h>
#include <gtk/gtk.h>
#include <gtk/gtkgl.h>
#include <libconfig.h++>

#include "spinSystem.h"
#include "renderSpins.h"
#include "paramgl.h"


// We'll create a very simple linked list of gradient parameter sets.
typedef struct _gradStruct
{
  float lDelta;
  float bDelta;
  float readOut;
  float gx;
  float gy;
  float gz;
  struct _gradStruct *next;
}gradStruct;
gradStruct *gGrads = NULL;

typedef struct _gtkInputStruct {

	GtkWidget *inputWindow;
	GtkWidget *inputSpinNumberField;
	GtkWidget *inputFieldStrengthField;
	GtkWidget *inputTimestepField;
	GtkWidget *inputOutsideAdcField;
	GtkWidget *inputOutsideT2Field;
	GtkWidget *inputMyelinAdcField;
	GtkWidget *inputMyelinT2Field;
	GtkWidget *inputInsideAdcField;
	GtkWidget *inputInsideT2Field;
	GtkWidget *inputPermeabilityField;
	GtkWidget *inputStepsPerUpdateField;
	GtkWidget *inputUseGpuRadioButton;
	GtkWidget *inputRenderOutputCheckButton;
	GtkWidget *inputUseRTreeRadioButton;
	GtkWidget *inputSimpleCollisionsRadioButton;
	GtkWidget *inputGradientFileChooser;
	GtkWidget *inputFiberFileChooser;

	uint *pNumSpins;
	float *pTimestep;
	float *pExtraAdc;
	float *pExtraT2;
	float *pMyelinAdc;
	float *pMyelinT2;
	float *pIntraAdc;
	float *pIntraT2;
	float *pPermeability;
	int *pStepsPerUpdate;
	bool *pUseGpu;
	bool *pUseDisplay;

	const char **sGradientFileName;
	const char **sFiberFileName;
} gtkInputStruct;

#define PI 3.14159265358979f
// view params
bool quitProgram = 0;

int ox, oy;
int buttonState = 0;
float camera_trans[] = {0, 0, -3};
float camera_rot[]   = {0, 0, 0};
float camera_trans_lag[] = {0, 0, -3};
float camera_rot_lag[] = {0, 0, 0};
const float inertia = 0.1;
float fiberAlpha = 0.3f;

FILE *outFilePtr;

bool renderSpins = true;
bool bPause = false;
bool displaySliders = false;
bool wireframe = false;
bool renderFibers = false;
bool renderSpheres = false;
bool isExit = false;
bool renderCubes = false;
float time_in_sec= 0.0f;

SpinRenderer::DisplayMode displayMode = SpinRenderer::SPIN_SPHERES;

//
// Simulation parameter defaults
//
// Most of these will be over-ridden by the configuration file.
//
float gyromagneticRatio = 42576.0f; // in KHz/T, which is equivalent to (cycles/millisecond)/T
//float larmorFreq = gyromagneticRatio * fieldStrength * 1e3f; // in Hz
float timestep = 0.005f; 		// in msec
float extraAdc = 2.0f;       		// micrometers^2/msec
float intraAdc = 2.0f;
float myelinAdc = 0.1;
float extraT2 = 80;;
float intraT2 = 80;
float myelinT2 = 7.5;
//float intraAdcScale = 1.0f;
int iterations = 1;
float permeability = -4.0f;
uint stepCount = 0;

SpinSystem *psystem = 0;
SpinSystem *psystemStill = 0;

// timesteps per second
time_t lastUpdateTime = 0;
float totalKernelTime = 0.0f;
float curStepsPerSec = 0.0f;

SpinRenderer *renderer = 0;

float modelView[16];

ParamListGL *params;

//
// Function: initGL() 
// Return type: void
// Parameters: NONE
// Description: This function initializes the OpenGL environment for GLUT window display
//

void initGL()
{  
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
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

float calculateFlops()
{
	int numOfOps = 246;
	return ((float)(numOfOps * psystem->getNumSpins() * iterations) /(time_in_sec * 1000000000.0f));
}

//
// Function: update()
// Return type: void
// Parameters: NONE
// Description: Function to update spin motion in a given time step
//

void update(){ 
    static int currentRun = 0;
    static float3 curG;
    float3 nullG = {0.0f, 0.0f, 0.0f};
    static int curBigDelta;
    static int curLittleDelta;
    static int curReadout;
    static int nUpdates = -1;
    static int curNumUpdates = 0;
    static bool done = false;
    static gradStruct *curGrad = gGrads;
    //printf("Starting diffusion update\n");
    //printf("m_spinRadius: %g \n", m_spinRadius);
   
	// update the simulation
	stepCount++;
	//printf("Inside update: stepCount=%i\n",stepCount);
	// A NULL gGrads indicates that we are running in demo-mode with no DW gradients 
	if(gGrads!=NULL){ 
      if(curNumUpdates<=nUpdates){
	      // Continue processing this DW run
	      if(curNumUpdates==0){
	         // Reset and then turn on dwGrads
	         
	         // We should only need to reset the phases to 0 here, which is faster 
	         // than a full system reset that also resets the random seeds and scrambles 
	         // spin positions. But skipping the full system reset causes a large bias
	         // to accumulate. We should track that down...
	         psystem->reset(SpinSystem::CONFIG_RANDOM);
	         psystem->setGradient(curG);
	         
	      }else if(curNumUpdates==curLittleDelta && curNumUpdates<curBigDelta){
	         // turn off DW grads
	         psystem->setGradient(nullG);
	      
	      }else if(curNumUpdates==curBigDelta){
	         psystem->setGradient(curG*-1.0f);
	         
	      }else if(curNumUpdates==curLittleDelta+curBigDelta){
	         // turn off DW grads
	         psystem->setGradient(nullG);	    
	              
	      }else if(curNumUpdates==curLittleDelta+curBigDelta+curReadout){
	         // measure MR signal and save it
		printf("Calling getMrSignal\n");
	         double mrSig = psystem->getMrSignal();
		printf("Done calling getMrSignal\n");
		      printf("*** MR signal=%0.6f ***\n", mrSig);
		      if(outFilePtr!=NULL){
		         fprintf(outFilePtr, "mrSig{m}(s) = %g;\n", mrSig);
		         fflush(outFilePtr);
		      }
	      }
	      curNumUpdates++;
      }else{
         // Load the params for the next DW run 
         curNumUpdates = 0;
         if(curGrad==NULL){
            done = true;
         }else{	    
            currentRun++;
            curG.x = curGrad->gx;
            curG.y = curGrad->gy;
            curG.z = curGrad->gz;
            printf("Running: d=%gms, D=%gms, readout=%gms, G=[%g,%g,%g]mT/m.\t",
                    curGrad->lDelta, curGrad->bDelta, curGrad->readOut, curGrad->gx, curGrad->gy, curGrad->gz);
            if(outFilePtr!=NULL){
                  fprintf(outFilePtr, "s = s+1;\n");
                  fprintf(outFilePtr, "delta{m}(s) = %g;\n", curGrad->lDelta);
                  fprintf(outFilePtr, "Delta{m}(s) = %g;\n", curGrad->bDelta);
                  fprintf(outFilePtr, "readOut{m}(s) = %g;\n", curGrad->readOut);
                  fprintf(outFilePtr, "dwGrads{m}(:,s) = [%g;%g;%g];\n", curG.x, curG.y, curG.z);
                  fflush(outFilePtr);
             }
             // Compute the total number of update steps needed for this run
             curBigDelta = (int)(curGrad->bDelta/(timestep*(float)iterations)+0.5);
             curLittleDelta = (int)(curGrad->lDelta/(timestep*(float)iterations)+0.5);
             curReadout = (int)(curGrad->readOut/(timestep*(float)iterations)+0.5);
   	         nUpdates = curBigDelta+curLittleDelta+curReadout;
   	         curGrad = curGrad->next;
         }
      }

      if(done){
		   printf("stepcount: %g \n",stepCount);
		   printf("All runs finished\n");
		   uint totalTimeSteps = psystem->getNumTimeSteps();
		   float stepsPerSec = (float)totalTimeSteps/totalKernelTime;
		   printf("\nTotal kernel time was %0.2f sec for %d timesteps (%0.4f steps/sec)\n", 
		            totalKernelTime, totalTimeSteps, stepsPerSec);
		   if(outFilePtr!=NULL){
              fprintf(outFilePtr, "%% Timing info:\n");
		      fprintf(outFilePtr, "stepsPerSec(m) = %g;\n", stepsPerSec);
              fprintf(outFilePtr, "totalTime(m) = %g;\n", totalKernelTime);
		      fclose(outFilePtr);
		   }
		   exit(0);
      }
	}
	//printf("Running update command\n");
	time_in_sec = psystem->update(timestep, iterations);
	//printf("Done running update command\n");
	curStepsPerSec = (float)iterations/time_in_sec;
	totalKernelTime += time_in_sec;
}

void DrawCube()
{

        glPushMatrix();
        glBegin(GL_POLYGON);

        //      This is the top face
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, -1.0f);
        glVertex3f(-1.0f, 0.0f, -1.0f);
        glVertex3f(-1.0f, 0.0f, 0.0f);

         //      This is the front face
         glVertex3f(0.0f, 0.0f, 0.0f);
         glVertex3f(-1.0f, 0.0f, 0.0f);
         glVertex3f(-1.0f, -1.0f, 0.0f);
         glVertex3f(0.0f, -1.0f, 0.0f);

         //      This is the right face
         glVertex3f(0.0f, 0.0f, 0.0f);
         glVertex3f(0.0f, -1.0f, 0.0f);
         glVertex3f(0.0f, -1.0f, -1.0f);
         glVertex3f(0.0f, 0.0f, -1.0f);

         //      This is the left face
         glVertex3f(-1.0f, 0.0f, 0.0f);
         glVertex3f(-1.0f, 0.0f, -1.0f);
         glVertex3f(-1.0f, -1.0f, -1.0f);
         glVertex3f(-1.0f, -1.0f, 0.0f);

          //      This is the bottom face
          glVertex3f(0.0f, 0.0f, 0.0f);
          glVertex3f(0.0f, -1.0f, -1.0f);
          glVertex3f(-1.0f, -1.0f, -1.0f);
          glVertex3f(-1.0f, -1.0f, 0.0f);

          //      This is the back face
          glVertex3f(0.0f, 0.0f, 0.0f);
          glVertex3f(-1.0f, 0.0f, -1.0f);
          glVertex3f(-1.0f, -1.0f, -1.0f);
          glVertex3f(0.0f, -1.0f, -1.0f);

          glEnd();
          glPopMatrix();
}
//
// Function: display()
// Return type: void
// Parameters: NONE
// Description: Function to render the spins/fibers etc on the GLUT window
//

void display()
{
    // update the simulation
	//printf("Starting display function!\n");

    if(!bPause){
        psystem->setAdc(extraAdc, intraAdc, myelinAdc);
        //psystem->setIntraAdcScale(intraAdcScale);
        psystem->setPermeability(pow(10,permeability));
    	
        update();
        renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumSpins());
        psystem->setColorFromSpin();
	//printf("psystem->getCurrentReadBuffer(): %u\n", psystem->getCurrentReadBuffer());
	//printf("psystem->getNumSpins(): %u\n", psystem->getNumSpins());
   }else{
         usleep(15000);
    }
    // render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  

    // view transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    for (int c = 0; c < 3; ++c)
    {
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
    }
    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

    // cube
    glColor3f(1.0, 1.0, 1.0);
    glutWireCube(2.0);
    

    if(renderSpins)
    {
	//printf("Testing renderSpins\n");
        renderer->display(displayMode);
    }


    if(renderFibers){
	//if (true){
       // fibers
	//printf("Testing renderFibers\n");
       glEnable(GL_BLEND);                 // Turn blending On
       glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);  // Simple transparency

       for(int i=0; i<psystem->getNumFibers(); i++){
          float *p = psystem->getFiberPos(i);
          // The squared radius is stored
          float radius = sqrt(p[3]);
          glPushMatrix();
          glColor4f(0.0, 0.3, 0.1, fiberAlpha);

          if(p[2]==2.0f){
             glTranslatef(p[0], p[1], -1.0f);
             glutSolidCylinder(radius, 2, 20, 1);
          }else if(p[1]==2.0f){
	          glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
             glTranslatef(p[0], p[2], -1.0f);
             glutSolidCylinder(radius, 2, 20, 1);
          }else if(p[0]==2.0f){
	          glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
             glTranslatef(p[1], p[2], -1.0f);
             glutSolidCylinder(radius, 2, 20, 1);
          }else{
              // It's a sphere
              glColor4f(0.0, 0.2, 1.0, fiberAlpha);
              glTranslatef(p[0], p[1], p[2]);
              glutSolidSphere(radius, 20, 20);
          }
          glPopMatrix();
       }       
       glDisable(GL_BLEND);
    }

    if(displaySliders) {
        glDisable(GL_DEPTH_TEST);
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
        glEnable(GL_BLEND);
        params->Render(0, 0);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
    }

    // This will draw the grid cubes that the space is grid into
    if(renderCubes) {
   	uint numCubes = psystem->getNumCubes();
   	float3 starting;
   	float gridLength = psystem->getCubeLength();
   	glEnable(GL_BLEND);                 // Turn blending On
      glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);  // Simple transparency

      starting.x = -1.0f + gridLength;
      for(uint i=0;i<numCubes;i++){
         starting.y = -1.0f + gridLength; 
         for(uint j=0;j<numCubes;j++){
            starting.z = -1.0f + gridLength;
            for(uint k=0;k<numCubes;k++){
           	   glPushMatrix();
               glColor4f(0.0, 0.2, 1.0, fiberAlpha);
               glTranslatef( starting.x, starting.y, starting.z);
               glScalef(gridLength, gridLength, gridLength);
               DrawCube();
               glPopMatrix();	       
               starting.z += gridLength;
            }
            starting.y += gridLength;		
         }  
         starting.x += gridLength;
      }
   	glDisable(GL_BLEND);
   }
	//printf("Right before glutSwapBuffers().\n");
	//std::cin.get();
   glutSwapBuffers();
	//printf("Done calling glutSwapBuffers().\n");
	//std::cin.get();
    
    // this displays the frame rate updated every second (independent of frame rate)
    if(time(NULL)>lastUpdateTime) {
        char msg[256];
        float mrSig = psystem->getMrSignal();
	printf("mrSig: %0.4f, ms: %4.2f \n", mrSig, (float)stepCount*timestep);
	//printf("stepCount, timestep: %i,%g\n", stepCount, timestep);
        sprintf(msg, "Dsim: %3.1f steps/sec, %4.1f ms, %0.4f MR sig, %0.4f GFlops", 
                         curStepsPerSec, (float)stepCount*timestep, mrSig, calculateFlops());  
        glutSetWindowTitle(msg);
        lastUpdateTime = time(NULL);
    }
    glutReportErrors();
	//printf("Finished with display function! \n");
}

void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 10.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    renderer->setWindowSize(w, h);
    renderer->setFOV(60.0);
}


void mouse(int button, int state, int x, int y)
{
    int mods;

    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    mods = glutGetModifiers();
    if (mods & GLUT_ACTIVE_SHIFT) {
        buttonState = 2;
    } else if (mods & GLUT_ACTIVE_CTRL) {
        buttonState = 3;
    }

    ox = x; oy = y;

    if (displaySliders) {
        if (params->Mouse(x, y, button, state)) {
            glutPostRedisplay();
            return;
        }
    }

    glutPostRedisplay();
}


// transform vector by matrix
void xform(float *v, float *r, GLfloat *m)
{
  r[0] = v[0]*m[0] + v[1]*m[4] + v[2]*m[8] + m[12];
  r[1] = v[0]*m[1] + v[1]*m[5] + v[2]*m[9] + m[13];
  r[2] = v[0]*m[2] + v[1]*m[6] + v[2]*m[10] + m[14];
}

// transform vector by transpose of matrix
void ixform(float *v, float *r, GLfloat *m)
{
  r[0] = v[0]*m[0] + v[1]*m[1] + v[2]*m[2];
  r[1] = v[0]*m[4] + v[1]*m[5] + v[2]*m[6];
  r[2] = v[0]*m[8] + v[1]*m[9] + v[2]*m[10];
}

void ixformPoint(float *v, float *r, GLfloat *m)
{
    float x[4];
    x[0] = v[0] - m[12];
    x[1] = v[1] - m[13];
    x[2] = v[2] - m[14];
    x[3] = 1.0f;
    ixform(x, r, m);
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - ox;
    dy = y - oy;

    if (displaySliders) {
        if (params->Motion(x, y)) {
            ox = x; oy = y;
            glutPostRedisplay();
            return;
        }
    }

    if (buttonState == 3) {
        // left+middle = zoom
        camera_trans[2] += (dy / 100.0) * 0.5 * fabs(camera_trans[2]);
    } 
    else if (buttonState & 2) {
        // middle = translate
        camera_trans[0] += dx / 100.0;
        camera_trans[1] -= dy / 100.0;
    }
    else if (buttonState & 1) {
        // left = rotate
        camera_rot[0] += dy / 5.0;
        camera_rot[1] += dx / 5.0;
    }

    ox = x; oy = y;
    glutPostRedisplay();
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

// commented out to remove unused parameter warnings in Linux 
void key(unsigned char key, int /*x*/, int /*y*/)
{

    switch (key) 
    {
    case ' ':
        bPause = !bPause;
        break;
    case 13:
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
	break;
    }

    glutPostRedisplay();
}

void special(int k, int x, int y)
{
    if (displaySliders) {
        params->Special(k, x, y);
    }
}

void idle(void)
{
    glutPostRedisplay();
}

void initParams()
{
    // create a new parameter list
    params = new ParamListGL("misc");
    params->AddParam(new Param<float>("time step (ms)", timestep, 0.001, 0.1, 0.0001, &timestep));
    params->AddParam(new Param<int>("iterations", iterations, 1, 100, 1, &iterations));
    params->AddParam(new Param<float>("Axon ADC (um^2/msec)", intraAdc, 0.0, 10.0, 0.01, &intraAdc));
    params->AddParam(new Param<float>("10^permeability", permeability, -9, 0, 0.1, &permeability));
}

void mainMenu(int i)
{
    key((unsigned char) i, 0, 0);
}

void initMenus()
{
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Reset block [1]", '1');
    glutAddMenuEntry("Reset random [2]", '2');
    glutAddMenuEntry("Toggle animation [ ]", ' ');
    glutAddMenuEntry("Step animation [ret]", 13);
    glutAddMenuEntry("Toggle fibers [t]", 't');
    glutAddMenuEntry("Toggle Spheres [s]", 's');  
    glutAddMenuEntry("Toggle sliders [h]", 'h');
    glutAddMenuEntry("Toggle grid cubes [c]", 'c');
    glutAddMenuEntry("Quit (esc)", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}


// Stop the GTK+ main loop function when the window is destroyed.
static void destroy (GtkWidget *window, gpointer data) {
	gtk_main_quit ();
}

// Return FALSE to destroy the widget. By returning TRUE, you can cancel a delete-event.
// This can be used to confirm quitting the application.
static gboolean delete_event (GtkWidget *window, GdkEvent *event, gpointer data) {
	quitProgram = 1;
	return FALSE;
}

// When a file is selected, display the full path in the GtkLabel widget.
static void gradient_file_changed (GtkFileChooser *fileChooser, GtkLabel *label) {

	gchar *file = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (fileChooser));
	gtk_label_set_text (label, file);
}

// When a file is selected, display the full path in the GtkLabel widget.
static void fiber_file_changed (GtkFileChooser *fileChooser, GtkLabel *label) {

	gchar *file = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (fileChooser));
	gtk_label_set_text (label, file);

	//const char *sFiberFileName = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (gtkInputs->inputFiberFileChooser));

	//FILE *fiberFilePtr  = fopen(fiberFileConst,"r");

	printf("file = %s\n", file);

	FILE *fiberFilePtr  = fopen(file,"r");

	if(fiberFilePtr){
		if(!psystemStill->initFibers(fiberFilePtr, 0.8))
			exit(-1);
		fclose(fiberFilePtr);
	}

	printf("Number of loaded fibers: %i\n", psystemStill->getNumFibers());
}

// Read the input fields and then kill the GUI when the "Run" button is clicked
static void runButton_clicked (GtkWidget *runButton, gtkInputStruct *gtkInputs) {
	//printf("Run button has been clicked!\n");

	//const gchar* spinNumberString;
	//spinNumberString = gtk_entry_get_text(GTK_ENTRY (gtkInputs->inputSpinNumberField));
	// *(gtkInputs->pNumSpins) = atoi(spinNumberString);

	*(gtkInputs->pNumSpins) = atoi(gtk_entry_get_text(GTK_ENTRY (gtkInputs->inputSpinNumberField)));
	*(gtkInputs->pTimestep) = atof(gtk_entry_get_text(GTK_ENTRY (gtkInputs->inputTimestepField)));
	*(gtkInputs->pExtraAdc) = atof(gtk_entry_get_text(GTK_ENTRY (gtkInputs->inputOutsideAdcField)));
	*(gtkInputs->pExtraT2) = atof(gtk_entry_get_text(GTK_ENTRY (gtkInputs->inputOutsideT2Field)));
	*(gtkInputs->pMyelinAdc) = atof(gtk_entry_get_text(GTK_ENTRY (gtkInputs->inputMyelinAdcField)));
	*(gtkInputs->pMyelinT2) = atof(gtk_entry_get_text(GTK_ENTRY (gtkInputs->inputMyelinT2Field)));
	*(gtkInputs->pIntraAdc) = atof(gtk_entry_get_text(GTK_ENTRY (gtkInputs->inputInsideAdcField)));
	*(gtkInputs->pIntraT2) = atof(gtk_entry_get_text(GTK_ENTRY (gtkInputs->inputInsideT2Field)));
	*(gtkInputs->pPermeability) = atof(gtk_entry_get_text(GTK_ENTRY (gtkInputs->inputPermeabilityField)));
	*(gtkInputs->pStepsPerUpdate) = atoi(gtk_entry_get_text(GTK_ENTRY (gtkInputs->inputStepsPerUpdateField)));
	*(gtkInputs->pUseGpu) = (bool) gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (gtkInputs->inputUseGpuRadioButton));
	*(gtkInputs->pUseDisplay) = (bool) gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (gtkInputs->inputRenderOutputCheckButton));

	*(gtkInputs->sGradientFileName) = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (gtkInputs->inputGradientFileChooser));
	*(gtkInputs->sFiberFileName) = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (gtkInputs->inputFiberFileChooser));

	// Destroy the GUI window
	gtk_widget_destroy (gtkInputs->inputWindow);

	// Break out of the GTK main loop
	gtk_main_quit ();
}


float boxv[][3] = {
	{ -0.5, -0.5, -0.5 },
	{  0.5, -0.5, -0.5 },
	{  0.5,  0.5, -0.5 },
	{ -0.5,  0.5, -0.5 },
	{ -0.5, -0.5,  0.5 },
	{  0.5, -0.5,  0.5 },
	{  0.5,  0.5,  0.5 },
	{ -0.5,  0.5,  0.5 }
};
#define ALPHA 0.5

static float ang = 30.;

/*
static gboolean
expose (GtkWidget *da, GdkEventExpose *event, gpointer user_data)
{

	GdkGLContext *glcontext = gtk_widget_get_gl_context (da);
	GdkGLDrawable *gldrawable = gtk_widget_get_gl_drawable (da);

	// g_print (" :: expose\n");

	if (!gdk_gl_drawable_gl_begin (gldrawable, glcontext))
	{
		g_assert_not_reached ();
	}

	// Draw in here
	// Note: Am temporarily trying to draw fibers, am mostly copying from display(), will clean up later

	// render
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  

	// view transform
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	for (int c = 0; c < 3; ++c)
	{
		camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
		camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
	}
	glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
	glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
	glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);


	glEnable(GL_BLEND);                 // Turn blending On
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);  // Simple transparency

	for(int i=0; i<psystemStill->getNumFibers(); i++){
		float *p = psystemStill->getFiberPos(i);
		// The squared radius is stored
		float radius = sqrt(p[3]);
		glPushMatrix();
		glColor4f(0.0, 0.3, 0.1, fiberAlpha);

		if(p[2]==2.0f){
			glTranslatef(p[0], p[1], -1.0f);
			glutSolidCylinder(radius, 2, 20, 1);
		}else if(p[1]==2.0f){
			glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
			glTranslatef(p[0], p[2], -1.0f);
			glutSolidCylinder(radius, 2, 20, 1);
		}else if(p[0]==2.0f){
			glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
			glTranslatef(p[1], p[2], -1.0f);
			glutSolidCylinder(radius, 2, 20, 1);
		}else{
			// It's a sphere
			glColor4f(0.0, 0.2, 1.0, fiberAlpha);
			glTranslatef(p[0], p[1], p[2]);
			glutSolidSphere(radius, 20, 20);
		}
		glPopMatrix();
	}
	glDisable(GL_BLEND);

   	//glutSwapBuffers();

	if (gdk_gl_drawable_is_double_buffered (gldrawable))
		gdk_gl_drawable_swap_buffers (gldrawable);

	else
		glFlush ();

	gdk_gl_drawable_gl_end (gldrawable);

	return TRUE;
}*/



static gboolean
expose (GtkWidget *da, GdkEventExpose *event, gpointer user_data)
{
	GdkGLContext *glcontext = gtk_widget_get_gl_context (da);
	GdkGLDrawable *gldrawable = gtk_widget_get_gl_drawable (da);

	// g_print (" :: expose\n");

	if (!gdk_gl_drawable_gl_begin (gldrawable, glcontext))
	{
		g_assert_not_reached ();
	}

	// Draw in here
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPushMatrix();
	
	glRotatef (ang, 1, 0, 1);
	// glRotatef (ang, 0, 1, 0);
	// glRotatef (ang, 0, 0, 1);

	glShadeModel(GL_FLAT);

#if 0
	glBegin (GL_QUADS);
	glColor4f(0.0, 0.0, 1.0, ALPHA);
	glVertex3fv(boxv[0]);
	glVertex3fv(boxv[1]);
	glVertex3fv(boxv[2]);
	glVertex3fv(boxv[3]);

	glColor4f(1.0, 1.0, 0.0, ALPHA);
	glVertex3fv(boxv[0]);
	glVertex3fv(boxv[4]);
	glVertex3fv(boxv[5]);
	glVertex3fv(boxv[1]);
	
	glColor4f(0.0, 1.0, 1.0, ALPHA);
	glVertex3fv(boxv[2]);
	glVertex3fv(boxv[6]);
	glVertex3fv(boxv[7]);
	glVertex3fv(boxv[3]);
	
	glColor4f(1.0, 0.0, 0.0, ALPHA);
	glVertex3fv(boxv[4]);
	glVertex3fv(boxv[5]);
	glVertex3fv(boxv[6]);
	glVertex3fv(boxv[7]);
	
	glColor4f(1.0, 0.0, 1.0, ALPHA);
	glVertex3fv(boxv[0]);
	glVertex3fv(boxv[3]);
	glVertex3fv(boxv[7]);
	glVertex3fv(boxv[4]);
	
	glColor4f(0.0, 1.0, 0.0, ALPHA);
	glVertex3fv(boxv[1]);
	glVertex3fv(boxv[5]);
	glVertex3fv(boxv[6]);
	glVertex3fv(boxv[2]);

	glEnd ();
#endif

	glBegin (GL_LINES);
	glColor3f (1., 0., 0.);
	glVertex3f (0., 0., 0.);
	glVertex3f (1., 0., 0.);
	glEnd ();
	
	glBegin (GL_LINES);
	glColor3f (0., 1., 0.);
	glVertex3f (0., 0., 0.);
	glVertex3f (0., 1., 0.);
	glEnd ();
	
	glBegin (GL_LINES);
	glColor3f (0., 0., 1.);
	glVertex3f (0., 0., 0.);
	glVertex3f (0., 0., 1.);
	glEnd ();

	glBegin(GL_LINES);
	glColor3f (1., 1., 1.);
	glVertex3fv(boxv[0]);
	glVertex3fv(boxv[1]);
	
	glVertex3fv(boxv[1]);
	glVertex3fv(boxv[2]);
	
	glVertex3fv(boxv[2]);
	glVertex3fv(boxv[3]);
	
	glVertex3fv(boxv[3]);
	glVertex3fv(boxv[0]);
	
	glVertex3fv(boxv[4]);
	glVertex3fv(boxv[5]);
	
	glVertex3fv(boxv[5]);
	glVertex3fv(boxv[6]);
	
	glVertex3fv(boxv[6]);
	glVertex3fv(boxv[7]);
	
	glVertex3fv(boxv[7]);
	glVertex3fv(boxv[4]);
	
	glVertex3fv(boxv[0]);
	glVertex3fv(boxv[4]);
	
	glVertex3fv(boxv[1]);
	glVertex3fv(boxv[5]);
	
	glVertex3fv(boxv[2]);
	glVertex3fv(boxv[6]);
	
	glVertex3fv(boxv[3]);
	glVertex3fv(boxv[7]);
	glEnd();

	glPopMatrix ();

	if (gdk_gl_drawable_is_double_buffered (gldrawable))
		gdk_gl_drawable_swap_buffers (gldrawable);

	else
		glFlush ();

	gdk_gl_drawable_gl_end (gldrawable);

	return TRUE;
}


static gboolean
configure (GtkWidget *da, GdkEventConfigure *event, gpointer user_data)
{
	GdkGLContext *glcontext = gtk_widget_get_gl_context (da);
	GdkGLDrawable *gldrawable = gtk_widget_get_gl_drawable (da);

	if (!gdk_gl_drawable_gl_begin (gldrawable, glcontext))
	{
		g_assert_not_reached ();
	}

	glLoadIdentity();
	glViewport (0, 0, da->allocation.width, da->allocation.height);
	glOrtho (-10,10,-10,10,-20050,10000);
	glEnable (GL_BLEND);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glScalef (10., 10., 10.);
	
	gdk_gl_drawable_gl_end (gldrawable);

	return TRUE;
}

static gboolean
rotate (gpointer user_data)
{
	GtkWidget *da = GTK_WIDGET (user_data);

	ang++;

	gdk_window_invalidate_rect (da->window, &da->allocation, FALSE);
	gdk_window_process_updates (da->window, FALSE);

	return TRUE;
}


////////////////////////////////////////////////////////////////////////////////
// Program main
//
// TO DO:
//   * Add command-line help text
//
////////////////////////////////////////////////////////////////////////////////
using namespace libconfig;
using namespace std;

int main(int argc, char* argv[]) {

	bool useGpu = true;
	bool useDisplay = true;
	uint numSpins = 20000;
	float spaceScale = 100.0f; // The native space is -1 to 1 um. Scale it up with this parameter.

	bool gradsFileGivenInConfigFile  = 0;
	bool gradsStrGivenInConfigFile = 0;
	bool fiberFileGivenInCommandLine = 0;
	bool fiberFileGivenInConfigFile = 0;
	float fiberRadius = 1.0f;
	float fiberRadiusStd = 0.1f;
	float fiberSpace = 0.2f;
	float fiberSpaceStd = 0.02f;
	float fiberInnerRadiusProportion = 0.8f;
	float crossFraction = 0.0f;
	psystemStill = new SpinSystem(100, 0, 20, gyromagneticRatio, 1);

	//const char *testGradientFileName = "/home/bragis/Academic/initialTestGradientFileName.grads";
	//const char *testFiberFileName = "/home/bragis/Academic/initialTestFiberFileName.fibers";

	char *configFilename;
	if(!cutGetCmdLineArgumentstr( argc, (const char**) argv, "config", &configFilename)) {
		configFilename = "sim.cfg";
	}

	// Load the configuration
	Config cfg;
	try{
		// Load the configuration..
		cout << "loading " << configFilename << "..." << endl;
		cfg.readFile(configFilename);
	}catch(FileIOException){
		cout << " File IO FAILED. " << endl;       
	}catch(ParseException){
		cout << " Parse FAILED. " << endl;
	}

	const char *gradsFile;
	if(cfg.lookupValue("sim.gradsFile", gradsFile)){
		gradsFileGivenInConfigFile = 1;
	} else if(cfg.lookupValue("sim.gradsStr", gradsFile)){
		gradsStrGivenInConfigFile = 1;
	}
	//testGradientFileName = gradsFile;

	const char *fiberFileConst;
	char *fiberFileFromCmdLine;
	if(cutGetCmdLineArgumentstr( argc, (const char**) argv, "fibersFile", &fiberFileFromCmdLine)){
	//if(cutGetCmdLineArgumentstr( argc, (const char**) argv, "fibersFile", &fiberFileConst)){
		fiberFileGivenInCommandLine = 1;
		fiberFileConst = fiberFileFromCmdLine;
	} else if(cfg.lookupValue("fibers.file", fiberFileConst)){
		fiberFileGivenInConfigFile = 1;
	}


	// App params
	cfg.lookupValue("app.stepsPerUpdate", iterations);

	cfg.lookupValue("app.useGpu", useGpu);
	// Allow command-line override:
	if(cutCheckCmdLineFlag(argc, (const char**) argv, "gpu")) useGpu = true;
	else if(cutCheckCmdLineFlag(argc, (const char**) argv, "cpu")) useGpu = false;

	cfg.lookupValue("app.useDisplay", useDisplay);
	// Allow command-line override:
	if(cutCheckCmdLineFlag(argc, (const char**) argv, "disp")) useDisplay = true;
	else if(cutCheckCmdLineFlag(argc, (const char**) argv, "nodisp")) useDisplay = false;

	// Read simulation parameters from config file
	cfg.lookupValue("sim.numSpins", numSpins);
	cfg.lookupValue("sim.gyromagneticRatio", gyromagneticRatio);
	cfg.lookupValue("sim.timestep", timestep);
	cfg.lookupValue("sim.extraAdc", extraAdc);
	cfg.lookupValue("sim.intraAdc", intraAdc);
	cfg.lookupValue("sim.myelinAdc", myelinAdc);
	cfg.lookupValue("sim.extraT2", extraT2);
	cfg.lookupValue("sim.intraT2", intraT2);
	cfg.lookupValue("sim.myelinT2", myelinT2);
	//cfg.lookupValue("sim.intraAdcScale", intraAdcScale);
	cfg.lookupValue("sim.spaceScale", spaceScale);
	cfg.lookupValue("sim.permeability", permeability);

	int cmdNumSpins = 0;
	if(cutGetCmdLineArgumenti( argc, (const char**) argv, "numSpins", &cmdNumSpins)){
		numSpins = (uint)abs(cmdNumSpins);
		printf("OVERRIDING number of spins with command-line arg: numSpins = %d\n", numSpins);
	}



	///////////////////////////////////
	// Start gtk GUI

	GtkWidget *window, *mainTable, *spinNumberLabel,  *spinNumberField, *fieldStrengthLabel, *fieldStrengthField;
	GtkWidget *timestepLabel, *timestepField, *tissueParameterLabel, *adcLabel, *t2Label, *betweenFibersLabel, *inMyelinLabel;
	GtkWidget *insideFibersLabel, *permeabilityLabel, *outsideAdcField, *outsideT2Field, *myelinAdcField, *myelinT2Field;
	GtkWidget *insideAdcField, *insideT2Field, *permeabilityField, *gradientFileChooser, *chosenGradientLabel, *gradientFileLabel;
	GtkWidget *chosenFibersLabel, *fiberFileLabel, *fiberFileChooser, *settingsTabs, *settingsLabel, *advancedLabel;
	GtkWidget *settingsTable, *advancedTable, *tempRenderingButton, *tempTimeEvalButton, *outputFileChooser;
	GtkWidget *stepsPerUpdateLabel, *useCpuRadioButton, *useGpuRadioButton, *renderOutputCheckButton, *stepsPerUpdateField;
	GtkWidget *useRTreeRadioButton, *useRectGridRadioButton, *allOutputLabel, *insideOutputLabel, *myelinOutputLabel;
	GtkWidget *outsideOutputLabel, *outputTabs, *lambda1AllLabel, *lambda2AllLabel, *lambda3AllLabel, *mdAllLabel, *faAllLabel;
	GtkWidget *rdAllLabel, *lambda1InsideLabel, *lambda2InsideLabel, *lambda3InsideLabel, *mdInsideLabel, *faInsideLabel, *rdInsideLabel;
	GtkWidget *lambda1MyelinLabel, *lambda2MyelinLabel, *lambda3MyelinLabel, *mdMyelinLabel, *faMyelinLabel, *rdMyelinLabel;
	GtkWidget *lambda1OutsideLabel, *lambda2OutsideLabel, *lambda3OutsideLabel, *mdOutsideLabel, *faOutsideLabel, *rdOutsideLabel;
	GtkWidget *allOutputTable, *insideOutputTable, *myelinOutputTable, *outsideOutputTable, *saveOutputButton, *runButton;
	GtkWidget *simpleCollisionsRadioButton, *advancedCollisionsRadioButton, *drawArea, *window2;
	GtkFileFilter *allFilesFilter, *gradientFilesFilter, *fiberFilesFilter;
	GdkGLConfig *glconfig;


	gtk_init (&argc, &argv);
	gtk_gl_init (&argc, &argv);

	// Create the drawing area for displaying GL
	drawArea = gtk_drawing_area_new ();

	// Will delete these
	tempRenderingButton = gtk_button_new_with_label ("Rendering of diffusion\nwill go here");
	tempTimeEvalButton = gtk_button_new_with_label ("Time evolution of gradient\nand signal\nwill go here");
	saveOutputButton = gtk_button_new_with_label ("Save output\nto m-file");
	runButton = gtk_button_new_with_label ("Run simulation");

	// Create the window
	window = gtk_window_new (GTK_WINDOW_TOPLEVEL);
	gtk_window_set_title (GTK_WINDOW (window), "Diffusion Simulator");
	gtk_container_set_border_width (GTK_CONTAINER (window), 10);
	gtk_widget_set_size_request (window, 1200, 550);

	window2 = gtk_window_new (GTK_WINDOW_TOPLEVEL);
	gtk_window_set_title (GTK_WINDOW (window2), "Diffusion Simulator GL");
	gtk_container_set_border_width (GTK_CONTAINER (window), 10);
	gtk_widget_set_size_request (window, 800, 550);

	// Create the tables
	mainTable = gtk_table_new (2,4,TRUE);
	settingsTable = gtk_table_new (15,4,TRUE);
	advancedTable = gtk_table_new (10,4,TRUE);
	allOutputTable = gtk_table_new (8,1,TRUE);
	insideOutputTable = gtk_table_new (8,1,TRUE);
	myelinOutputTable = gtk_table_new (8,1,TRUE);
	outsideOutputTable = gtk_table_new (8,1,TRUE);

	// Create notebooks for settings and output
	settingsTabs = gtk_notebook_new ();
	settingsLabel = gtk_label_new ("Settings");
	advancedLabel = gtk_label_new ("Advanced");
	outputTabs = gtk_notebook_new ();
	allOutputLabel = gtk_label_new ("All");
	insideOutputLabel = gtk_label_new ("Inside");
	myelinOutputLabel = gtk_label_new ("Myelin");
	outsideOutputLabel = gtk_label_new ("Outside");

	// Create callback functions for clicking of notebook
	//g_signal_connect (G_OBJECT (settingsTable), "clicked",
	//		  G_CALLBACK (switch_page),
	//		  (gpointer) settingsTabs);
	//g_signal_connect (G_OBJECT (advancedTable), "clicked",
	//		  G_CALLBACK (switch_page),
	//		  (gpointer) settingsTabs);


	// Append to pages to the notebook containers.
	gtk_notebook_append_page (GTK_NOTEBOOK (settingsTabs), settingsTable, settingsLabel);
	gtk_notebook_append_page (GTK_NOTEBOOK (settingsTabs), advancedTable, advancedLabel);
	gtk_notebook_set_tab_pos (GTK_NOTEBOOK (settingsTabs), GTK_POS_TOP);

	gtk_notebook_append_page (GTK_NOTEBOOK (outputTabs), allOutputTable, allOutputLabel);
	gtk_notebook_append_page (GTK_NOTEBOOK (outputTabs), insideOutputTable, insideOutputLabel);
	gtk_notebook_append_page (GTK_NOTEBOOK (outputTabs), myelinOutputTable, myelinOutputLabel);
	gtk_notebook_append_page (GTK_NOTEBOOK (outputTabs), outsideOutputTable, outsideOutputLabel);
	gtk_notebook_set_tab_pos (GTK_NOTEBOOK (outputTabs), GTK_POS_TOP);


	// Create labels for basic settings
	spinNumberLabel = gtk_label_new ("Number of spins: ");
	fieldStrengthLabel = gtk_label_new ("Field strength (T): ");
	timestepLabel = gtk_label_new ("Time step: ");
	tissueParameterLabel = gtk_label_new ("Tissue parameters");
	adcLabel = gtk_label_new ("ADC");
	t2Label = gtk_label_new ("T2");
	betweenFibersLabel = gtk_label_new ("Between fibers");
	inMyelinLabel = gtk_label_new ("In myelin");
	insideFibersLabel = gtk_label_new ("Inside fibers");
	permeabilityLabel = gtk_label_new ("Permeability");
	chosenGradientLabel = gtk_label_new("Gradient file:");
	gradientFileLabel = gtk_label_new ("");
	chosenFibersLabel = gtk_label_new("Fiber file:");
	fiberFileLabel = gtk_label_new ("");

	// Create labels for advanced settings
	stepsPerUpdateLabel = gtk_label_new ("Steps per update: ");

	// Create labels for output tabs
	lambda1AllLabel = gtk_label_new ("\u03BB1: ");
	lambda2AllLabel = gtk_label_new ("\u03BB2: ");
	lambda3AllLabel = gtk_label_new ("\u03BB3: ");
	lambda1InsideLabel = gtk_label_new ("\u03BB1: ");
	lambda2InsideLabel = gtk_label_new ("\u03BB2: ");
	lambda3InsideLabel = gtk_label_new ("\u03BB3: ");
	lambda1MyelinLabel = gtk_label_new ("\u03BB1: ");
	lambda2MyelinLabel = gtk_label_new ("\u03BB2: ");
	lambda3MyelinLabel = gtk_label_new ("\u03BB3: ");
	lambda1OutsideLabel = gtk_label_new ("\u03BB1: ");
	lambda2OutsideLabel = gtk_label_new ("\u03BB2: ");
	lambda3OutsideLabel = gtk_label_new ("\u03BB3: ");
	mdAllLabel = gtk_label_new ("md: ");
	faAllLabel = gtk_label_new ("fa: ");
	rdAllLabel = gtk_label_new ("rd: ");
	mdInsideLabel = gtk_label_new ("md: ");
	faInsideLabel = gtk_label_new ("fa: ");
	rdInsideLabel = gtk_label_new ("rd: ");
	mdMyelinLabel = gtk_label_new ("md: ");
	faMyelinLabel = gtk_label_new ("fa: ");
	rdMyelinLabel = gtk_label_new ("rd: ");
	mdOutsideLabel = gtk_label_new ("md: ");
	faOutsideLabel = gtk_label_new ("fa: ");
	rdOutsideLabel = gtk_label_new ("rd: ");

	// Create entry fields for basic settings
	char tempEntryStr [30];
	spinNumberField = gtk_entry_new ();
	gtk_entry_set_width_chars (GTK_ENTRY (spinNumberField), 6);
	//gtk_entry_set_text (GTK_ENTRY (spinNumberField), "20000");
	sprintf(tempEntryStr,"%d",numSpins);
	gtk_entry_set_text (GTK_ENTRY (spinNumberField), tempEntryStr);
	// Note: Does the field strength matter at all?
	fieldStrengthField = gtk_entry_new ();
	gtk_entry_set_width_chars (GTK_ENTRY (fieldStrengthField), 6);
	gtk_entry_set_text (GTK_ENTRY (fieldStrengthField), "1.5");
	timestepField = gtk_entry_new ();
	gtk_entry_set_width_chars (GTK_ENTRY (timestepField), 6);
	//gtk_entry_set_text (GTK_ENTRY (timestepField), "0.005");
	sprintf(tempEntryStr,"%.3f",timestep);
	gtk_entry_set_text (GTK_ENTRY (timestepField), tempEntryStr);
	outsideAdcField = gtk_entry_new ();
	gtk_entry_set_width_chars (GTK_ENTRY (outsideAdcField), 6);
	//gtk_entry_set_text (GTK_ENTRY (outsideAdcField), "2.1");
	sprintf(tempEntryStr,"%.1f",extraAdc);
	gtk_entry_set_text (GTK_ENTRY (outsideAdcField), tempEntryStr);
	outsideT2Field = gtk_entry_new ();
	gtk_entry_set_width_chars (GTK_ENTRY (outsideT2Field), 6);
	//gtk_entry_set_text (GTK_ENTRY (outsideT2Field), "80");
	sprintf(tempEntryStr,"%.1f",extraT2);
	gtk_entry_set_text (GTK_ENTRY (outsideT2Field), tempEntryStr);
	myelinAdcField = gtk_entry_new ();
	gtk_entry_set_width_chars (GTK_ENTRY (myelinAdcField), 6);
	//gtk_entry_set_text (GTK_ENTRY (myelinAdcField), "0.1");
	sprintf(tempEntryStr,"%.1f",myelinAdc);
	gtk_entry_set_text (GTK_ENTRY (myelinAdcField), tempEntryStr);
	myelinT2Field = gtk_entry_new ();
	gtk_entry_set_width_chars (GTK_ENTRY (myelinT2Field), 6);
	//gtk_entry_set_text (GTK_ENTRY (myelinT2Field), "7.5");
	sprintf(tempEntryStr,"%.1f",myelinT2);
	gtk_entry_set_text (GTK_ENTRY (myelinT2Field), tempEntryStr);
	insideAdcField = gtk_entry_new ();
	gtk_entry_set_width_chars (GTK_ENTRY (insideAdcField), 6);
	//gtk_entry_set_text (GTK_ENTRY (insideAdcField), "2.1");
	sprintf(tempEntryStr,"%.1f",intraAdc);
	gtk_entry_set_text (GTK_ENTRY (insideAdcField), tempEntryStr);
	insideT2Field = gtk_entry_new ();
	gtk_entry_set_width_chars (GTK_ENTRY (insideT2Field), 6);
	//gtk_entry_set_text (GTK_ENTRY (insideT2Field), "80");
	sprintf(tempEntryStr,"%.1f",intraT2);
	gtk_entry_set_text (GTK_ENTRY (insideT2Field), tempEntryStr);
	permeabilityField = gtk_entry_new ();
	gtk_entry_set_width_chars (GTK_ENTRY (permeabilityField), 6);
	//gtk_entry_set_text (GTK_ENTRY (permeabilityField), "-6");
	sprintf(tempEntryStr,"%.1f",permeability);
	gtk_entry_set_text (GTK_ENTRY (permeabilityField), tempEntryStr);

	// Create entry fields for advanced settings
	stepsPerUpdateField = gtk_entry_new ();
	gtk_entry_set_width_chars (GTK_ENTRY (stepsPerUpdateField), 6);
	//gtk_entry_set_text (GTK_ENTRY (stepsPerUpdateField), "10");
	sprintf(tempEntryStr,"%i",iterations);
	gtk_entry_set_text(GTK_ENTRY (stepsPerUpdateField), tempEntryStr);

	// Create file chooser buttons
	gradientFileChooser = gtk_file_chooser_button_new ("Choose gradient file", GTK_FILE_CHOOSER_ACTION_OPEN);
	gtk_file_chooser_button_set_width_chars(GTK_FILE_CHOOSER_BUTTON (gradientFileChooser), 10);
	gtk_file_chooser_set_filename (GTK_FILE_CHOOSER (gradientFileChooser), gradsFile);
	fiberFileChooser = gtk_file_chooser_button_new ("Choose fiber file", GTK_FILE_CHOOSER_ACTION_OPEN);
	gtk_file_chooser_button_set_width_chars(GTK_FILE_CHOOSER_BUTTON (fiberFileChooser), 10);
	gtk_file_chooser_set_filename (GTK_FILE_CHOOSER (fiberFileChooser), fiberFileConst);

	// Create output file chooser
	//outputFileChooser = gtk_file_chooser_button_new ("Choose file for saving output", GTK_FILE_CHOOSER_ACTION_SAVE);
	//gtk_file_chooser_button_set_width_chars(GTK_FILE_CHOOSER_BUTTON (outputFileChooser), 10);
	//gtk_file_chooser_set_filename (GTK_FILE_CHOOSER (outputFileChooser), "/output.m");
	

	// Monitor when the selected file is changed.
	g_signal_connect (G_OBJECT (gradientFileChooser), "selection_changed", G_CALLBACK (gradient_file_changed), (gpointer) gradientFileLabel);
	g_signal_connect (G_OBJECT (fiberFileChooser), "selection_changed", G_CALLBACK (fiber_file_changed), (gpointer) fiberFileLabel);

	// Set file chooser buttons to the location of the user's home directory.
	//gtk_file_chooser_set_current_folder (GTK_FILE_CHOOSER (gradientFileChooser), g_get_home_dir());
	//gtk_file_chooser_set_current_folder (GTK_FILE_CHOOSER (fiberFileChooser), g_get_home_dir());

	// Provide a filter to show all files and one to show only 3 types of images.
	allFilesFilter = gtk_file_filter_new ();
	gtk_file_filter_set_name (allFilesFilter, "All files");
	gtk_file_filter_add_pattern (allFilesFilter, "*");
	gradientFilesFilter = gtk_file_filter_new ();
	gtk_file_filter_set_name (gradientFilesFilter, "Gradient files");
	gtk_file_filter_add_pattern (gradientFilesFilter, "*.grads");
	fiberFilesFilter = gtk_file_filter_new ();
	gtk_file_filter_set_name (fiberFilesFilter, "Fiber files");
	gtk_file_filter_add_pattern (fiberFilesFilter, "*.fibers");

	// Add filters to the file chooser buttons
	gtk_file_chooser_add_filter (GTK_FILE_CHOOSER (gradientFileChooser), gradientFilesFilter);
	gtk_file_chooser_add_filter (GTK_FILE_CHOOSER (gradientFileChooser), allFilesFilter);
	gtk_file_chooser_add_filter (GTK_FILE_CHOOSER (fiberFileChooser), fiberFilesFilter);
	gtk_file_chooser_add_filter (GTK_FILE_CHOOSER (fiberFileChooser), allFilesFilter);


	// Create buttons for the advanced settings tab
	useCpuRadioButton = gtk_radio_button_new_with_label (NULL, "Use CPU");
	useGpuRadioButton = gtk_radio_button_new_with_label_from_widget (GTK_RADIO_BUTTON (useCpuRadioButton), "Use GPU");
	renderOutputCheckButton = gtk_check_button_new_with_label ("Render output");
	gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON (renderOutputCheckButton), TRUE);
	useRTreeRadioButton = gtk_radio_button_new_with_label (NULL, "Use R-Tree");
	useRectGridRadioButton = gtk_radio_button_new_with_label_from_widget (GTK_RADIO_BUTTON (useRTreeRadioButton), "Use grid");
	simpleCollisionsRadioButton = gtk_radio_button_new_with_label(NULL, "Simple collisions");
	advancedCollisionsRadioButton = gtk_radio_button_new_with_label_from_widget (GTK_RADIO_BUTTON (simpleCollisionsRadioButton), "Advanced collisions");


	// Create the table for the basic settings - belongs to a notebook tab
	gtk_table_attach (GTK_TABLE (settingsTable), spinNumberLabel, 0, 2, 1, 2,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), spinNumberField, 2, 3, 1, 2,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), fieldStrengthLabel, 0, 2, 2, 3,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), fieldStrengthField, 2, 3, 2, 3,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), timestepLabel, 0, 2, 3, 4,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), timestepField, 2, 3, 3, 4,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), tissueParameterLabel, 0, 4, 5, 6,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), adcLabel, 2, 3, 6, 7,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), t2Label, 3, 4, 6, 7,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), betweenFibersLabel, 0, 2, 7, 8,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), outsideAdcField, 2, 3, 7, 8,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), outsideT2Field, 3, 4, 7, 8,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), inMyelinLabel, 0, 2, 8, 9,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), myelinAdcField, 2, 3, 8, 9,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), myelinT2Field, 3, 4, 8, 9,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), insideFibersLabel, 0, 2, 9, 10,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), insideAdcField, 2, 3, 9, 10,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), insideT2Field, 3, 4, 9, 10,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), permeabilityLabel, 0, 2, 11, 12,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), permeabilityField, 2, 3, 11, 12,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), chosenGradientLabel, 0, 2, 13, 14,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), gradientFileChooser, 2, 4, 13, 14,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), chosenFibersLabel, 0, 2, 14, 15,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (settingsTable), fiberFileChooser, 2, 4, 14, 15,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);


	// Create the table for the advanced settings - belongs to a notebook tab
	gtk_table_attach (GTK_TABLE (advancedTable), useCpuRadioButton, 0, 3, 1, 2,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (advancedTable), useGpuRadioButton, 0, 3, 2, 3,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (advancedTable), renderOutputCheckButton, 0, 3, 4, 5,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (advancedTable), stepsPerUpdateLabel, 0, 2, 6, 7,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (advancedTable), stepsPerUpdateField, 2, 4, 6, 7,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (advancedTable), useRTreeRadioButton, 0, 3, 8, 9,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (advancedTable), useRectGridRadioButton, 0, 3, 9, 10,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (advancedTable), simpleCollisionsRadioButton, 0, 3, 11, 12,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (advancedTable), advancedCollisionsRadioButton, 0, 3, 12, 13,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	

	// Create the table for the output - belongs to a notebook tab
	gtk_table_attach (GTK_TABLE (allOutputTable), lambda1AllLabel, 1, 3, 1, 2,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (allOutputTable), lambda2AllLabel, 1, 3, 2, 3,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (allOutputTable), lambda3AllLabel, 1, 3, 3, 4,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (allOutputTable), mdAllLabel, 1, 3, 4, 5,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (allOutputTable), faAllLabel, 1, 3, 5, 6,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (allOutputTable), rdAllLabel, 1, 3, 6, 7,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);

	gtk_table_attach (GTK_TABLE (insideOutputTable), lambda1InsideLabel, 1, 3, 1, 2,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (insideOutputTable), lambda2InsideLabel, 1, 3, 2, 3,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (insideOutputTable), lambda3InsideLabel, 1, 3, 3, 4,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (insideOutputTable), mdInsideLabel, 1, 3, 4, 5,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (insideOutputTable), faInsideLabel, 1, 3, 5, 6,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (insideOutputTable), rdInsideLabel, 1, 3, 6, 7,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);

	gtk_table_attach (GTK_TABLE (myelinOutputTable), lambda1MyelinLabel, 1, 3, 1, 2,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (myelinOutputTable), lambda2MyelinLabel, 1, 3, 2, 3,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (myelinOutputTable), lambda3MyelinLabel, 1, 3, 3, 4,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (myelinOutputTable), mdMyelinLabel, 1, 3, 4, 5,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (myelinOutputTable), faMyelinLabel, 1, 3, 5, 6,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (myelinOutputTable), rdMyelinLabel, 1, 3, 6, 7,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);

	gtk_table_attach (GTK_TABLE (outsideOutputTable), lambda1OutsideLabel, 1, 3, 1, 2,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (outsideOutputTable), lambda2OutsideLabel, 1, 3, 2, 3,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (outsideOutputTable), lambda3OutsideLabel, 1, 3, 3, 4,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (outsideOutputTable), mdOutsideLabel, 1, 3, 4, 5,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (outsideOutputTable), faOutsideLabel, 1, 3, 5, 6,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_table_attach (GTK_TABLE (outsideOutputTable), rdOutsideLabel, 1, 3, 6, 7,
			  GTK_SHRINK, GTK_SHRINK, 0, 0);
	

	// Attach notebooks and buttons to the main table
	gtk_table_attach (GTK_TABLE (mainTable), settingsTabs, 0, 1, 0, 4,
			  GTK_SHRINK, GTK_EXPAND, 0, 0);
	gtk_table_attach (GTK_TABLE (mainTable), tempRenderingButton, 1, 3, 0, 2,
			  GTK_EXPAND, GTK_EXPAND, 0, 0);
	//gtk_table_attach (GTK_TABLE (mainTable), drawArea, 1, 3, 0, 2,
	//		  GTK_EXPAND, GTK_EXPAND, 0, 0);
	gtk_table_attach (GTK_TABLE (mainTable), tempTimeEvalButton, 1, 3, 2, 4,
			  GTK_EXPAND, GTK_EXPAND, 0, 0);
	gtk_table_attach (GTK_TABLE (mainTable), outputTabs, 3, 4, 0, 2,
			  GTK_SHRINK, GTK_EXPAND, 0, 0);
	gtk_table_attach (GTK_TABLE (mainTable), saveOutputButton, 3, 4, 2, 3,
			  GTK_EXPAND, GTK_EXPAND, 0, 0);
	gtk_table_attach (GTK_TABLE (mainTable), runButton, 3, 4, 3, 4,
			  GTK_EXPAND, GTK_EXPAND, 0, 0);
	

	// Add five pixels of spacing between every row and every column.
	gtk_table_set_row_spacings (GTK_TABLE (mainTable), 5);
	gtk_table_set_col_spacings (GTK_TABLE (mainTable), 5);
	gtk_table_set_row_spacings (GTK_TABLE (settingsTable), 5);
	gtk_table_set_col_spacings (GTK_TABLE (settingsTable), 5);
	gtk_table_set_row_spacings (GTK_TABLE (allOutputTable), 5);
	gtk_table_set_col_spacings (GTK_TABLE (allOutputTable), 5);
	gtk_table_set_row_spacings (GTK_TABLE (insideOutputTable), 5);
	gtk_table_set_col_spacings (GTK_TABLE (insideOutputTable), 5);
	gtk_table_set_row_spacings (GTK_TABLE (myelinOutputTable), 5);
	gtk_table_set_col_spacings (GTK_TABLE (myelinOutputTable), 5);
	gtk_table_set_row_spacings (GTK_TABLE (outsideOutputTable), 5);
	gtk_table_set_col_spacings (GTK_TABLE (outsideOutputTable), 5);

	// Create an "input structure" - this is necessary because the callback function
	// for the "run" button accepts a limited number of arguments.
	gtkInputStruct *gtkInputs = (gtkInputStruct*) malloc(sizeof(gtkInputStruct));
	gtkInputs->inputWindow = window;
	gtkInputs->inputSpinNumberField = spinNumberField;
	gtkInputs->inputFieldStrengthField = fieldStrengthField;
	gtkInputs->inputTimestepField = timestepField;
	gtkInputs->inputOutsideAdcField = outsideAdcField;
	gtkInputs->inputOutsideT2Field = outsideT2Field;
	gtkInputs->inputMyelinAdcField = myelinAdcField;
	gtkInputs->inputMyelinT2Field = myelinT2Field;
	gtkInputs->inputInsideAdcField = insideAdcField;
	gtkInputs->inputInsideT2Field = insideT2Field;
	gtkInputs->inputPermeabilityField = permeabilityField;
	gtkInputs->inputStepsPerUpdateField = stepsPerUpdateField;
	gtkInputs->inputUseGpuRadioButton = useGpuRadioButton;
	gtkInputs->inputRenderOutputCheckButton = renderOutputCheckButton;
	gtkInputs->inputUseRTreeRadioButton = useRTreeRadioButton;
	gtkInputs->inputSimpleCollisionsRadioButton = simpleCollisionsRadioButton;
	gtkInputs->pNumSpins = &numSpins;
	gtkInputs->pTimestep = &timestep;
	gtkInputs->pExtraAdc = &extraAdc;
	gtkInputs->pExtraT2 = &extraT2;
	gtkInputs->pMyelinAdc = &myelinAdc;
	gtkInputs->pMyelinT2 = &myelinT2;
	gtkInputs->pIntraAdc = &intraAdc;
	gtkInputs->pIntraT2 = &intraT2;
	gtkInputs->pPermeability = &permeability;
	gtkInputs->pUseGpu = &useGpu;
	gtkInputs->pUseDisplay = &useDisplay;
	gtkInputs->pStepsPerUpdate = &iterations;
	gtkInputs->inputGradientFileChooser = gradientFileChooser;
	gtkInputs->inputFiberFileChooser = fiberFileChooser;
	//gtkInputs->sGradientFileName = &testGradientFileName;
	gtkInputs->sGradientFileName = &gradsFile;
	//gtkInputs->sFiberFileName = &testFiberFileName;
	gtkInputs->sFiberFileName = &fiberFileConst;

	// Connect buttons to callback functions
	g_signal_connect (G_OBJECT (runButton), "clicked", G_CALLBACK (runButton_clicked), (gpointer) gtkInputs);

	// Connect the main window to the destroy and delete-event signals.
	g_signal_connect (G_OBJECT (window), "destroy", G_CALLBACK (destroy), NULL);
	g_signal_connect (G_OBJECT (window), "delete_event", G_CALLBACK (delete_event), NULL);

	gtk_container_add (GTK_CONTAINER (window), mainTable);
	gtk_container_add (GTK_CONTAINER (window2), drawArea);

	gtk_widget_set_events (drawArea, GDK_EXPOSURE_MASK);

	gtk_widget_show (window);
	gtk_widget_show (window2);
	//gtk_widget_show (drawArea);

	glconfig = gdk_gl_config_new_by_mode (static_cast<GdkGLConfigMode>
			(GDK_GL_MODE_RGB |
			GDK_GL_MODE_DEPTH |
			GDK_GL_MODE_DOUBLE) );

	if (!glconfig)
	{
		g_assert_not_reached ();
	}
	
	if (!gtk_widget_set_gl_capability (drawArea, glconfig, NULL, TRUE,
				GDK_GL_RGBA_TYPE))
	{
		g_assert_not_reached ();
	}

	g_signal_connect (drawArea, "configure-event",
			G_CALLBACK (configure), NULL);
	g_signal_connect (drawArea, "expose-event",
			G_CALLBACK (expose), NULL);

	gtk_widget_show_all (window);
	gtk_widget_show_all (window2);

	//g_timeout_add (1000 / 60, rotate, drawArea);

	gtk_main ();

	printf("Info read from GUI:");
	printf("numSpins = %i\n",numSpins);
	printf("timestep = %g\n", timestep);
	printf("extraAdc = %g\n", extraAdc);
	printf("extraT2 = %g\n", extraT2);
	printf("myelinAdc = %g\n", myelinAdc);
	printf("myelinT2 = %g\n", myelinT2);
	printf("intraAdc = %g\n", intraAdc);
	printf("intraT2 = %g\n", intraT2);
	printf("permeability = %g\n", permeability);
	printf("useGpu = %i\n", useGpu);
	printf("useDisplay = %i\n", useDisplay);
	printf("iterations = %i\n", iterations);
	printf("gradient file = %s\n", gradsFile);
	printf("fiber file = %s\n", fiberFileConst);
	//exit(0);

	// End of gtk GUI
	/////////////////////////////////////////////////////////////

	if (quitProgram){
		exit(1);
	}



    printf("*\nStarting main diffusion run.\n*\n");

    if (gradsFileGivenInConfigFile){
        printf("Loading gradient sequence from %s.\n", gradsFile);
        FILE *gradsFilePtr = fopen(gradsFile,"r");
        float ld,bd,ro,gx,gy,gz;
        gradStruct *prevGrad=NULL, *curGrad=NULL;
        while(!feof(gradsFilePtr)){
            int nScan = fscanf(gradsFilePtr,"%g %g %g %g %g %g", &ld, &bd, &ro, &gx, &gy, &gz);
            if(nScan==6){
                curGrad = new gradStruct;
                curGrad->lDelta = ld;
                curGrad->bDelta = bd;
                curGrad->readOut = ro;
                curGrad->gx = gx;
                curGrad->gy = gy;
                curGrad->gz = gz;
                curGrad->next = NULL;
                if(prevGrad!=NULL) prevGrad->next = curGrad;
                if(gGrads==NULL) gGrads = curGrad;
                prevGrad = curGrad;
            }
        }
        fclose(gradsFilePtr);
        //curGrad = gGrads;
        //while(curGrad!=NULL){
        //    printf("Initializing: d=%gms, D=%gms, readout=%gms, G=[%g,%g,%g]mT/m.\n", 
        //            curGrad->lDelta, curGrad->bDelta, curGrad->readOut, curGrad->gx, curGrad->gy, curGrad->gz);
        //    curGrad = curGrad->next;
        //}
	//}else if(cfg.lookupValue("sim.gradsStr", gradsFile)){
	}else if (gradsStrGivenInConfigFile){
		float ld,bd,ro,gx,gy,gz;
		int nScan = sscanf(gradsFile,"%g %g %g %g %g %g", &ld, &bd, &ro, &gx, &gy, &gz);
		if(nScan==6){
			gGrads = new gradStruct;
			gGrads->lDelta = ld;
			gGrads->bDelta = bd;
			gGrads->readOut = ro;
			gGrads->gx = gx;
			gGrads->gy = gy;
			gGrads->gz = gz;
			gGrads->next = NULL;
		}
	}

	// Fiber params
	float cmdVal = NAN;
	FILE *fiberFilePtr;
	//char *fiberFile;
	if (fiberFileGivenInCommandLine || fiberFileGivenInConfigFile){
		//fiberFile = (char *)malloc((strlen(fiberFileConst)+1)*sizeof(char));
		//strcpy(fiberFile, fiberFileConst);
		printf("Reading fibers from file (%s)\n",fiberFileConst);
		fiberFilePtr = fopen(fiberFileConst,"r");	
		//fiberFilePtr = fopen(fiberFile,"r");
		//if (fiberFileGivenInCommandLine){
		//	printf("Reading fibers from file (%s)\n",fiberFile);
		//	fiberFilePtr = fopen(fiberFile,"r");
		//} else if(fiberFileGivenInConfigFile){
		//	printf("Reading fibers from file (%s)\n",fiberFile);
		//	fiberFilePtr = fopen(fiberFile,"r");
	} else{
		cfg.lookupValue("fibers.radiusMean", fiberRadius);
		cfg.lookupValue("fibers.radiusStdev", fiberRadiusStd);
		cfg.lookupValue("fibers.spacingMean", fiberSpace);
		cfg.lookupValue("fibers.spacingStdev", fiberSpaceStd);
		cfg.lookupValue("fibers.crossFraction", crossFraction);
    
		if(cutGetCmdLineArgumentf( argc, (const char**) argv, "radiusMean", &cmdVal)){
			// Keep the std proportional to the mean
			fiberRadiusStd = fiberRadiusStd/fiberRadius*cmdVal;
			fiberRadius = cmdVal;
			printf("OVERRIDING mean fiber radius with command-line arg: mean = %0.3g, std = %0.4g\n", fiberRadius, fiberRadiusStd);
		}

		if(cutGetCmdLineArgumentf( argc, (const char**) argv, "spaceMean", &cmdVal)){
			fiberSpace = cmdVal;
			// Keep the std proportional to the mean
			fiberSpaceStd = fiberSpaceStd/fiberSpace*cmdVal;
			fiberSpace = cmdVal;
			printf("OVERRIDING mean fiber spacing with command-line arg: mean = %0.3g, std = %0.4g\n", fiberSpace, fiberSpaceStd);
		}
	}

	if(cutGetCmdLineArgumentf( argc, (const char**) argv, "spaceScale", &cmdVal)){
		spaceScale = cmdVal;
		printf("OVERRIDING space scale with command-line arg: spaceScale = %0.1f\n", spaceScale);
	}

	if(cutGetCmdLineArgumentf( argc, (const char**) argv, "crossFraction", &cmdVal)){
		crossFraction = cmdVal;
		printf("OVERRIDING proportion of crossing fibers with command-line arg: crossFraction = %g\n", crossFraction);
	}

	cfg.lookupValue("fibers.innerRadiusFraction", fiberInnerRadiusProportion);


	char *outFilename;
	if(cutGetCmdLineArgumentstr( argc, (const char**) argv, "out", &outFilename)){
		outFilePtr = fopen(outFilename,"at");
		time_t now = time(NULL);
		fprintf(outFilePtr,"\n%% * * * %s", ctime(&now));
		fprintf(outFilePtr,"if(exist('m','var')), m=m+1; else, m=1; end\n");
		fprintf(outFilePtr,"s=0;\n");        
		fprintf(outFilePtr,"radiusMean(m) = %g;\n", fiberRadius);
		fprintf(outFilePtr,"radiusSpace(m) = %g;\n", fiberSpace);
		fprintf(outFilePtr,"spaceScale(m) = %g;\n", spaceScale);
		fprintf(outFilePtr,"numSpins(m) = %d;\n", numSpins);
		fprintf(outFilePtr,"crossFraction(m) = %g;\n", crossFraction);
		fprintf(outFilePtr,"useGpu(m) = %d;\n", useGpu);
		//fprintf(outFilePtr,"fibersFile{m} = '%s';\n", fiberFile);
		fprintf(outFilePtr,"fibersFile{m} = '%s';\n", fiberFileConst);
	}else{
		outFilePtr = NULL;
	}

	if(useDisplay){
		// if the display is enabled, then the spin system will keep spin
		// positions in a vbo array for faster OpenGL updates. So, we need to make sure
		// the GLUT window is prepared before we do anything with the spin system.
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
		glutInitWindowSize(640, 480);
		glutCreateWindow("Diffusion Simulator");
		initGL();
	}

	if(useGpu) printf("Executing simulation on GPU\n");
	else printf("executing simulation on CPU\n");

	// Create a default spin system
	psystem = new SpinSystem(numSpins, useGpu, spaceScale, gyromagneticRatio, useDisplay); 
	psystem->setAdc(extraAdc,intraAdc,myelinAdc);
	psystem->setT2(extraT2,intraT2,myelinT2);
	psystem->setPermeability(pow(10,permeability));

	// Initialize the fiber entites
	if(fiberFilePtr){
		if(!psystem->initFibers(fiberFilePtr, fiberInnerRadiusProportion))
			exit(-1);
		fclose(fiberFilePtr);
	}else{
		if(!psystem->initFibers(fiberRadius, fiberRadiusStd, fiberSpace, fiberSpaceStd, fiberInnerRadiusProportion, crossFraction)) 
			exit(-1);
	}

	// Build the spin system
	printf("*\nBuilding the spin system\n*\n");
	if(!psystem->build()) exit(-1); 

	// Randomize the spins 
	printf("*\nRandomizing the spins\n*\n");
	psystem->reset(SpinSystem::CONFIG_RANDOM);
	printf("*\nFinished randomizing the spins\n*\n");

	//psystem->printOutCells();

	//float testCalc = sqrt(5.0);
	//printf("Square root of 5: %g\n", testCalc);

	// Do an update to initalize the GPU stuff
	printf("*\nStarting system update\n*\n");
	printf("Before initial update: timestep = %g\n", timestep);
	psystem->update(timestep, 1);
	printf("After initial update: timestep = %g\n", timestep);
	printf("*\nFinished with initial update\n*\n");
	printf("*\nuseDisplay: %u\n", useDisplay);

	if(useDisplay){

		renderer = new SpinRenderer;
		renderer->setSpinRadius(psystem->getSpinRadius());
		renderer->setColorBuffer(psystem->getColorBuffer()); 
		initParams();
		initMenus();
		glutDisplayFunc(display);
		glutReshapeFunc(reshape);
		glutMouseFunc(mouse);
		glutMotionFunc(motion);
		glutKeyboardFunc(key);
		glutSpecialFunc(special);
		glutIdleFunc(idle);
		glutMainLoop();

	}else{
		// Just run the updates without rendering the spins
		while(1){    
			update();
		}
	}
	if (psystem) delete psystem;

	return 0;

}
