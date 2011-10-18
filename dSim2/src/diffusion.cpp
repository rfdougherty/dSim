// Example code from http://wiki.wxwidgets.org/WxGLCanvas
//
// To compile: g++ main.cpp -o test `wx-config --libs --cxxflags --gl-libs`

// Start includes from dSim
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
// End includes from dSim

#include "wx/wx.h"
#include "wx/sizer.h"
#include "wx/glcanvas.h"
#include "diffusion.h"
 
// include OpenGL
#ifdef __WXMAC__
#include "OpenGL/glu.h"
#include "OpenGL/gl.h"
#else
#include <GL/glu.h>
#include <GL/gl.h>
#endif

#include <libconfig.h++>


// Start typedef from dSim
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
// End typedef from dSim

class SettingsFrame: public wxFrame
{
public:

    SettingsFrame(const wxString& title, const wxPoint& pos, const wxSize& size);

    wxButton *okButton;
    wxButton *cancelButton;
    wxStaticText *spinNumberText;
    wxTextCtrl *spinNumberEntry;
    wxStaticText *timeStepText;
    wxTextCtrl *timeStepEntry;
    wxStaticText *ADCText;
    wxStaticText *T2Text;
    wxStaticText *betweenFibersText;
    wxTextCtrl *betweenFibersADCEntry;
    wxTextCtrl *betweenFibersT2Entry;
    wxStaticText *inMyelinText;
    wxTextCtrl *inMyelinADCEntry;
    wxTextCtrl *inMyelinT2Entry;
    wxStaticText *insideFibersText;
    wxTextCtrl *insideFibersADCEntry;
    wxTextCtrl *insideFibersT2Entry;
    wxStaticText *permeabilityText;
    wxTextCtrl *permeabilityEntry;

    void OnExit(wxCommandEvent& event);

    DECLARE_EVENT_TABLE() 
};

class MyFrame: public wxFrame
{
public:

    MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size);

    SettingsFrame *basicSettingsFrame;

    void OnQuit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);
    void OnRun(wxCommandEvent& event);
    void OnBasicSettings(wxCommandEvent& event);
    void OnAdvancedSettings(wxCommandEvent& event);
    void OnLoadSettings(wxCommandEvent& event);
    void OnLoadFibers(wxCommandEvent& event);
    void OnLoadGrads(wxCommandEvent& event);

    DECLARE_EVENT_TABLE()
};


class AdvancedSettingsFrame: public wxFrame
{
public:

    AdvancedSettingsFrame(const wxString& title, const wxPoint& pos, const wxSize& size);

    wxButton *okButton;
    wxButton *cancelButton;
    wxStaticText *stepsPerUpdateText;
    wxTextCtrl *stepsPerUpdateEntry;

    wxCheckBox *renderSpinsCheckbox;
    wxRadioButton *useGpuRadioButton;
    wxRadioButton *useCpuRadioButton;
    wxRadioButton *useRTreeRadioButton;
    wxRadioButton *useGridRadioButton;
    wxRadioButton *simpleCollisionsRadioButton;
    wxRadioButton *advancedCollisionsRadioButton;

    void OnExit(wxCommandEvent& event);

    DECLARE_EVENT_TABLE() 
};

enum
{
    ID_Quit = 1,
    ID_About,
    ID_Run,
    ID_BasicSettings,
    ID_AdvancedSettings,
    ID_LoadSettings,
    ID_LoadFibers,
    ID_LoadGrads,
    ID_BUTTON_Exit,
};

BEGIN_EVENT_TABLE(MyFrame, wxFrame)
    EVT_MENU(ID_Run, MyFrame::OnRun)
    EVT_MENU(ID_Quit, MyFrame::OnQuit)
    EVT_MENU(ID_About, MyFrame::OnAbout)
    EVT_MENU(ID_BasicSettings, MyFrame::OnBasicSettings)
    EVT_MENU(ID_AdvancedSettings, MyFrame::OnAdvancedSettings)
    EVT_MENU(ID_LoadSettings, MyFrame::OnLoadSettings)
    EVT_MENU(ID_LoadFibers, MyFrame::OnLoadFibers)
    EVT_MENU(ID_LoadGrads, MyFrame::OnLoadGrads)
END_EVENT_TABLE()

BEGIN_EVENT_TABLE(SettingsFrame, wxFrame)
    EVT_BUTTON(ID_BUTTON_Exit, SettingsFrame::OnExit)
END_EVENT_TABLE()

BEGIN_EVENT_TABLE(AdvancedSettingsFrame, wxFrame)
    EVT_BUTTON(ID_BUTTON_Exit, AdvancedSettingsFrame::OnExit)
END_EVENT_TABLE()


class MyApp: public wxApp
{
    virtual bool OnInit();
    
    MyFrame *frame2;
    wxFrame *frame;
    //SettingsFrame *basicSettingsFrame;
    AdvancedSettingsFrame *myAdvancedSettingsFrame;
    BasicGLPane * glPane;
public:
    
};

// Start dSim definitions

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
// End dSim definitions

 
IMPLEMENT_APP(MyApp)
 
 
bool MyApp::OnInit()
{
    wxBoxSizer* sizer = new wxBoxSizer(wxHORIZONTAL);
    //frame = new wxFrame((wxFrame *)NULL, -1,  wxT("wxWidgets with GL"), wxPoint(450,450), wxSize(400,200));

    frame2 = new MyFrame( _("dSim"), wxPoint(50, 50), wxSize(800,600) );
    //basicSettingsFrame = new SettingsFrame( _("Basic Settings"), wxPoint(100,100), wxSize(400,400) );
    myAdvancedSettingsFrame = new AdvancedSettingsFrame( _("Advanced Settings"), wxPoint(600,100), wxSize(400,400) );
	
    int args[] = {WX_GL_RGBA, WX_GL_DOUBLEBUFFER, WX_GL_DEPTH_SIZE, 16, 0};
    
    glPane = new BasicGLPane( (wxFrame*) frame2, args);
    sizer->Add(glPane, 1, wxEXPAND);

    frame2->SetSizer(sizer);
    frame2->SetAutoLayout(true);
    frame2->Show();

    //basicSettingsFrame->Show();
    myAdvancedSettingsFrame->Show();

    //frame->SetSizer(sizer);
    //frame->SetAutoLayout(true);
	
    //frame->Show();
    return true;
} 
 
BEGIN_EVENT_TABLE(BasicGLPane, wxGLCanvas)
EVT_MOTION(BasicGLPane::mouseMoved)
EVT_LEFT_DOWN(BasicGLPane::mouseDown)
EVT_LEFT_UP(BasicGLPane::mouseReleased)
EVT_RIGHT_DOWN(BasicGLPane::rightClick)
EVT_LEAVE_WINDOW(BasicGLPane::mouseLeftWindow)
EVT_SIZE(BasicGLPane::resized)
EVT_KEY_DOWN(BasicGLPane::keyPressed)
EVT_KEY_UP(BasicGLPane::keyReleased)
EVT_MOUSEWHEEL(BasicGLPane::mouseWheelMoved)
EVT_PAINT(BasicGLPane::render)
END_EVENT_TABLE()
 
 
// some useful events to use
void BasicGLPane::mouseMoved(wxMouseEvent& event) {}
void BasicGLPane::mouseDown(wxMouseEvent& event) {}
void BasicGLPane::mouseWheelMoved(wxMouseEvent& event) {}
void BasicGLPane::mouseReleased(wxMouseEvent& event) {}
void BasicGLPane::rightClick(wxMouseEvent& event) {}
void BasicGLPane::mouseLeftWindow(wxMouseEvent& event) {}
void BasicGLPane::keyPressed(wxKeyEvent& event) {}
void BasicGLPane::keyReleased(wxKeyEvent& event) {}
 
// Vertices and faces of a simple cube to demonstrate 3D render
// source: http://www.opengl.org/resources/code/samples/glut_examples/examples/cube.c
GLfloat v[8][3];
GLint faces[6][4] = {  /* Vertex indices for the 6 faces of a cube. */
    {0, 1, 2, 3}, {3, 2, 6, 7}, {7, 6, 5, 4},
    {4, 5, 1, 0}, {5, 6, 2, 1}, {7, 4, 0, 3} };



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

/*
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
*/

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
void key(unsigned char key, int 
    //x
     , int 
    //y
    )
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

int main(int argc, char* argv[]) {
*/
using namespace libconfig;
using namespace std;

void mainDiffusion(int argc, char* argv[]){
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
        printf("Number of spins: %i \n", numSpins);
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


    // Start wxwidgets
        
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

	//useDisplay = 0;		// This is temporary - some bug with the display
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
        printf("numSpins right before initializing psystem: %i\n", numSpins);
	psystem = new SpinSystem(numSpins, useGpu, spaceScale, gyromagneticRatio, useDisplay); 
	psystem->setAdc(extraAdc,intraAdc,myelinAdc);
	psystem->setT2(extraT2,intraT2,myelinT2);
	psystem->setPermeability(pow(10,permeability));

	// Initialize the fiber entities
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
    
	//return 0;

}

////////////////////////////////////////////////////////////////////////////////
// Program main
//
// TO DO:
//   * Add command-line help text
//
////////////////////////////////////////////////////////////////////////////////
//using namespace libconfig;
//using namespace std;


BasicGLPane::BasicGLPane(wxFrame* parent, int* args) :
    wxGLCanvas(parent, wxID_ANY, args, wxDefaultPosition, wxDefaultSize, wxFULL_REPAINT_ON_RESIZE)
{
	m_context = new wxGLContext(this);
    // prepare a simple cube to demonstrate 3D render
    // source: http://www.opengl.org/resources/code/samples/glut_examples/examples/cube.c
    v[0][0] = v[1][0] = v[2][0] = v[3][0] = -1;
    v[4][0] = v[5][0] = v[6][0] = v[7][0] = 1;
    v[0][1] = v[1][1] = v[4][1] = v[5][1] = -1;
    v[2][1] = v[3][1] = v[6][1] = v[7][1] = 1;
    v[0][2] = v[3][2] = v[4][2] = v[7][2] = 1;
    v[1][2] = v[2][2] = v[5][2] = v[6][2] = -1;    
 
    // To avoid flashing on MSW
    SetBackgroundStyle(wxBG_STYLE_CUSTOM);
}
 
BasicGLPane::~BasicGLPane()
{
	delete m_context;
}
 
void BasicGLPane::resized(wxSizeEvent& evt)
{
//	wxGLCanvas::OnSize(evt);
	
    Refresh();
}
 
/** Inits the OpenGL viewport for drawing in 3D. */
void BasicGLPane::prepare3DViewport(int topleft_x, int topleft_y, int bottomrigth_x, int bottomrigth_y)
{
	
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black Background
    glClearDepth(1.0f);	// Depth Buffer Setup
    glEnable(GL_DEPTH_TEST); // Enables Depth Testing
    glDepthFunc(GL_LEQUAL); // The Type Of Depth Testing To Do
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	
    glEnable(GL_COLOR_MATERIAL);
	
    glViewport(topleft_x, topleft_y, bottomrigth_x-topleft_x, bottomrigth_y-topleft_y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	
    float ratio_w_h = (float)(bottomrigth_x-topleft_x)/(float)(bottomrigth_y-topleft_y);
    gluPerspective(45 /*view angle*/, ratio_w_h, 0.1 /*clip close*/, 200 /*clip far*/);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
	
}
 
/** Inits the OpenGL viewport for drawing in 2D. */
void BasicGLPane::prepare2DViewport(int topleft_x, int topleft_y, int bottomrigth_x, int bottomrigth_y)
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black Background
    glEnable(GL_TEXTURE_2D);   // textures
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	
    glViewport(topleft_x, topleft_y, bottomrigth_x-topleft_x, bottomrigth_y-topleft_y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    gluOrtho2D(topleft_x, bottomrigth_x, bottomrigth_y, topleft_y);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}
 
int BasicGLPane::getWidth()
{
    return GetSize().x;
}
 
int BasicGLPane::getHeight()
{
    return GetSize().y;
}
 
 
void BasicGLPane::render( wxPaintEvent& evt )
{
    if(!IsShown()) return;
    
    wxGLCanvas::SetCurrent(*m_context);
    wxPaintDC(this); // only to be used in paint events. use wxClientDC to paint outside the paint event
	
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
    // ------------- draw some 2D ----------------
    prepare2DViewport(0,0,getWidth()/2, getHeight());
    glLoadIdentity();
	
    // white background
    glColor4f(1, 1, 1, 1);
    glBegin(GL_QUADS);
    glVertex3f(0,0,0);
    glVertex3f(getWidth(),0,0);
    glVertex3f(getWidth(),getHeight(),0);
    glVertex3f(0,getHeight(),0);
    glEnd();
	
    // red square
    glColor4f(1, 0, 0, 1);
    glBegin(GL_QUADS);
    glVertex3f(getWidth()/8, getHeight()/3, 0);
    glVertex3f(getWidth()*3/8, getHeight()/3, 0);
    glVertex3f(getWidth()*3/8, getHeight()*2/3, 0);
    glVertex3f(getWidth()/8, getHeight()*2/3, 0);
    glEnd();
    
    // ------------- draw some 3D ----------------

    prepare3DViewport(getWidth()/2,0,getWidth(), getHeight());
    glLoadIdentity();
	
    glColor4f(0,0,1,1);
    glTranslatef(0,0,-5);
    glRotatef(50.0f, 0.0f, 1.0f, 0.0f);
    
    glColor4f(1, 0, 0, 1);
    for (int i = 0; i < 6; i++)
    {
        glBegin(GL_LINE_STRIP);
        glVertex3fv(&v[faces[i][0]][0]);
        glVertex3fv(&v[faces[i][1]][0]);
        glVertex3fv(&v[faces[i][2]][0]);
        glVertex3fv(&v[faces[i][3]][0]);
        glVertex3fv(&v[faces[i][0]][0]);
        glEnd();
    }
    
    glFlush();
    SwapBuffers();
}


MyFrame::MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size)
: wxFrame( NULL, -1, title, pos, size )
{
    SettingsFrame *basicSettingsFrame;
    basicSettingsFrame = new SettingsFrame( _("Basic Settings"), wxPoint(100,100), wxSize(400,400) );

    basicSettingsFrame->Show();

    wxMenu *menuFile = new wxMenu;
    wxMenu *menuSettings = new wxMenu;

    menuFile->Append( ID_Run, _("&Run") );
    menuFile->AppendSeparator();
    menuFile->Append( ID_About, _("&About...") );
    menuFile->AppendSeparator();
    menuFile->Append( ID_Quit, _("E&xit") );

    menuSettings->Append( ID_BasicSettings, _("&Basic") );
    menuSettings->Append( ID_AdvancedSettings, _("&Advanced") );
    menuSettings->AppendSeparator();
    menuSettings->Append( ID_LoadSettings, _("Load &settings file") );
    menuSettings->Append( ID_LoadFibers, _("Load &fiber file") );
    menuSettings->Append( ID_LoadGrads, _("Load &gradient file") );

    wxMenuBar *menuBar = new wxMenuBar;
    menuBar->Append( menuFile, _("&File") );
    menuBar->Append( menuSettings, _("&Settings") );

    SetMenuBar( menuBar );

    CreateStatusBar();
    SetStatusText( _("This will contain some helpful explanation message") );
}



void MyFrame::OnQuit(wxCommandEvent& WXUNUSED(event))
{
    Close(TRUE);
}

void MyFrame::OnAbout(wxCommandEvent& WXUNUSED(event))
{
    wxMessageBox( _("Program information:\ndSim (Diffusion Simulator) is a program to simulate the diffusion of spins within a volume of fibers, measured within one TR of an MRI sequence. Developed by Bob Dougherty and Bragi Sveinsson."),
                  _("About dSim"),
                  wxOK | wxICON_INFORMATION, this);
}

void MyFrame::OnRun(wxCommandEvent& WXUNUSED(event))
{
  // Here we would start the run of the program
  printf("In OnRun function... \n");
  int argc = 1;
  char *argv;
  mainDiffusion(argc,&argv);
}

void MyFrame::OnBasicSettings(wxCommandEvent& WXUNUSED(event))
{
    /*wxMessageBox( _("Program information:\ndSim (Diffusion Simulator) is a program to simulate the diffusion of spins within a volume of fibers, measured within one TR of an MRI sequence. Developed by Bob Dougherty and Bragi Sveinsson."),
                  _("Basic Settings"),
                  wxOK | wxICON_INFORMATION, this);*/
    //basicSettingsFrame->Show();
}

void MyFrame::OnAdvancedSettings(wxCommandEvent& WXUNUSED(event))
{
    wxMessageBox( _("Program information:\ndSim (Diffusion Simulator) is a program to simulate the diffusion of spins within a volume of fibers, measured within one TR of an MRI sequence. Developed by Bob Dougherty and Bragi Sveinsson."),
                  _("Advanced Settings"),
                  wxOK | wxICON_INFORMATION, this);
}

void MyFrame::OnLoadSettings(wxCommandEvent& WXUNUSED(event))
{
	wxString settingsFile = wxFileSelector(_("Load settings file"),_(""),_("sim.cfg"),_(""),_("cfg files (*.cfg)|*.*|All files (*.*)|*.*"));
}

void MyFrame::OnLoadFibers(wxCommandEvent& WXUNUSED(event))
{
	wxString fiberFile = wxFileSelector(_("Load fiber file"),_(""),_("sim.fibers"),_(""),_("fiber files (*.fibers)|*.*|All files (*.*)|*.*"));
}

void MyFrame::OnLoadGrads(wxCommandEvent& WXUNUSED(event))
{
	wxString gradsFile = wxFileSelector(_("Load gradient file"),_(""),_("sim.grads"),_(""),_("grads files (*.grads)|*.*|All files (*.*)|*.*"));
}

SettingsFrame::SettingsFrame(const wxString& title, const wxPoint& pos, const wxSize& size)
: wxFrame( NULL, -1, title, pos, size )
{

    okButton = new wxButton(this, ID_BUTTON_Exit, _("OK"), wxPoint(200,330), wxDefaultSize, 0);
    cancelButton = new wxButton(this, ID_BUTTON_Exit, _("Cancel"), wxPoint(300,330), wxDefaultSize, 0);

    spinNumberText = new wxStaticText(this, -1, _("Number of spins:"), wxPoint(50,35), wxDefaultSize,0);
    spinNumberEntry = new wxTextCtrl(this, -1, _("100"), wxPoint(200,30), wxDefaultSize, 0);
    timeStepText = new wxStaticText(this, -1, _("Time step:"), wxPoint(50,65), wxDefaultSize,0);
    timeStepEntry = new wxTextCtrl(this, -1, _("0.005"), wxPoint(200,60), wxDefaultSize, 0);

    ADCText = new wxStaticText(this, -1, _("ADC"), wxPoint(200,120), wxDefaultSize,0);
    T2Text = new wxStaticText(this, -1, _("T2"), wxPoint(300,120), wxDefaultSize,0);

    betweenFibersText = new wxStaticText(this, -1, _("Between fibers:"), wxPoint(50,145), wxDefaultSize,0);
    betweenFibersADCEntry = new wxTextCtrl(this, -1, _("2.1"), wxPoint(200,140), wxDefaultSize, 0);
    betweenFibersT2Entry = new wxTextCtrl(this, -1, _("80.0"), wxPoint(300,140), wxDefaultSize, 0);

    inMyelinText = new wxStaticText(this, -1, _("In myelin:"), wxPoint(50,175), wxDefaultSize,0);
    inMyelinADCEntry = new wxTextCtrl(this, -1, _("0.1"), wxPoint(200,170), wxDefaultSize, 0);
    inMyelinT2Entry = new wxTextCtrl(this, -1, _("7.5"), wxPoint(300,170), wxDefaultSize, 0);

    insideFibersText = new wxStaticText(this, -1, _("Inside fibers:"), wxPoint(50,205), wxDefaultSize,0);
    insideFibersADCEntry = new wxTextCtrl(this, -1, _("2.1"), wxPoint(200,200), wxDefaultSize, 0);
    insideFibersT2Entry = new wxTextCtrl(this, -1, _("80.0"), wxPoint(300,200), wxDefaultSize, 0);

    permeabilityText = new wxStaticText(this, -1, _("Permeability:"), wxPoint(50,245), wxDefaultSize,0);
    permeabilityEntry = new wxTextCtrl(this, -1, _("-6.0"), wxPoint(200,240), wxDefaultSize, 0);

    CreateStatusBar();
    SetStatusText( _("This will contain some helpful explanation message") );
}

void SettingsFrame::OnExit(wxCommandEvent& WXUNUSED(event))
{
    Close(TRUE);
}


AdvancedSettingsFrame::AdvancedSettingsFrame(const wxString& title, const wxPoint& pos, const wxSize& size)
: wxFrame( NULL, -1, title, pos, size )
{
    okButton = new wxButton(this, ID_BUTTON_Exit, _("OK"), wxPoint(200,330), wxDefaultSize, 0);
    cancelButton = new wxButton(this, ID_BUTTON_Exit, _("Cancel"), wxPoint(300,330), wxDefaultSize, 0);

    stepsPerUpdateText = new wxStaticText(this, -1, _("Steps per update:"), wxPoint(50,35), wxDefaultSize,0);
    stepsPerUpdateEntry = new wxTextCtrl(this, -1, _("10"), wxPoint(200,30), wxDefaultSize, 0);

    renderSpinsCheckbox = new wxCheckBox(this, -1, _("Render spins"), wxPoint(50,65), wxDefaultSize,0);

    useGpuRadioButton = new wxRadioButton(this, -1, _("Use GPU"), wxPoint(50,100), wxDefaultSize, wxRB_GROUP);
    useCpuRadioButton = new wxRadioButton(this, -1, _("Use CPU"), wxPoint(50,130), wxDefaultSize);

    useRTreeRadioButton = new wxRadioButton(this, -1, _("Use R-Tree"), wxPoint(50,170), wxDefaultSize,wxRB_GROUP);
    useGridRadioButton = new wxRadioButton(this, -1, _("Use grid"), wxPoint(50,200), wxDefaultSize);

    simpleCollisionsRadioButton = new wxRadioButton(this, -1, _("Simple collisions"), wxPoint(50,250), wxDefaultSize, wxRB_GROUP);
    advancedCollisionsRadioButton = new wxRadioButton(this, -1, _("Advanced collisions"), wxPoint(50,280), wxDefaultSize);

    CreateStatusBar();
    SetStatusText( _("This will contain some helpful explanation message") );
}

void AdvancedSettingsFrame::OnExit(wxCommandEvent& WXUNUSED(event))
{
    Close(TRUE);
}
