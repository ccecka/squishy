#ifndef OPENGLVIEWER_H
#define OPENGLVIEWER_H

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// GL includes
#include <GL/glew.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// CUDA GL include
#include <cuda_gl_interop.h>

// Other includes
#include "../MySolver/Solver.h"
#include "../MyMatrix/Matrix.h"

#include "Vec3.h"
#include "Camera.h"


// Shortcut for Buffer Offset
#define BUF_OFF(i) ((char*)NULL + (i))

// Vertex Buffer Object
int nDim;
int nNodes;
GLuint vbo;
void* d_vbo;     // CUDA Vertex Buffer Object

// Index Buffer Object
GLuint ibo;
int ibo_size;

// Window dimensions
unsigned int winW = 1024, winH = 768;

// Mouse and keyboard controls
int mouse_old_x, mouse_old_y;
int mouse_state = 0;
const static int MOUSE_STATE_LEFT   = (1 << GLUT_LEFT_BUTTON);
const static int MOUSE_STATE_RIGHT  = (1 << GLUT_RIGHT_BUTTON);
const static int MOUSE_STATE_BOTH   = (MOUSE_STATE_LEFT | MOUSE_STATE_RIGHT);
const static int MOUSE_STATE_MIDDLE = (1 << GLUT_MIDDLE_BUTTON);
int record_key = 0;
int idle_on_key = 1; //pause and unpause

// Camera
GL_Camera camera;

// Recording
int image_num = 1; //used to name output files

// Surface Nodes
vector<unsigned int> surfaceNodes;

// Force Matrix
matrix<MY_REAL> forceMatrix;

// Solver
Solver<MY_REAL>* solver;

struct Keynode
{
  // Distance cut off for adjacent nodes
  const static float adj_max_distSq = 0.25 * 0.25;
  // Display radius
  const static float radius = 0.03;
  
  // List of "adjacent nodes" to the keynode and the scale of the force
  list< pair<int,float> > adjNodes;

  int keynode;

  // Construct a keynode as the nth node in the Mesh
  Keynode( int keynode_, const matrix<MY_REAL>& coord )
  : keynode(keynode_)
  {
    double nX = coord(keynode,0), nY = coord(keynode,1), nZ = coord(keynode,2);
    double mag = 1000;
    double c = -3.0 / adj_max_distSq;

    for( int n = 0; n < nNodes; ++n ) {
      double dx = coord(n,0) - nX, dy = coord(n,1) - nY, dz = coord(n,2) - nZ;
      double distSq = dx*dx + dy*dy + dz*dz;

      if( distSq < adj_max_distSq ) {
	float alpha = mag * exp( c * distSq );
	adjNodes.push_back( make_pair(n,alpha) );
      }
    }
  }

  // A keynode is really just a global node index
  inline operator int () const { return keynode; }

  inline void drawKeynode( float R, float G, float B )
  {
    glPointSize(50);
    glColor3f(R, G, B);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    //glVertexPointer(3, GL_FLOAT, 3*sizeof(float), BUF_OFF(0));
    glVertexPointer(nDim, GL_REAL, 0, BUF_OFF(nDim*keynode*sizeof(GL_REAL)));
    glEnableClientState(GL_VERTEX_ARRAY);
    
    glDrawArrays(GL_POINTS, 0, 1);
  }

  inline static void drawKeynode( float x, float y, float z,
				  float R, float G, float B )
  {
    // Could draw from VBO?
    glPushMatrix();
    glColor3f(R,G,B);
    glTranslatef(x,y,z);
    glutSolidSphere(radius,10,10);
      //glutSolidSphere(sqrt(adj_max_distSq),10,10);
    glPopMatrix();
  }
};


// List of keynodes
vector<Keynode> keynodeList;
const static int NULL_NODE = -1;
int active_keynode = NULL_NODE;


/*****************************/
/****** Helper Functions *****/
/*****************************/

/* Saves the current frame
CAUTION: With a 1024X768 window, each frame is ~9 MB.
         With a 640x480 window, each frame is ~3 MB.
*/
void saveFrame()
{
  // Read pixels into vector
  vector<char> pixels(3*winW*winH);
  glReadPixels(0, 0, winW, winH, GL_RGB, GL_BYTE, &pixels[0]);
 
  // Create corresponding filename
  int zero_digits = 7 - (int) ceil( log10( image_num+1 ) );

  string filename = "myFile";
  for( int d = 0; d < zero_digits; ++d ) filename += "0";
  filename += toString(image_num) + ".bin";
  
  // Write to binary file
  cerr << "Saving " << filename << endl;
  writeBIN(pixels,filename.c_str());
  
  // Increase image_num
  ++image_num;
}

// Defines a line   x(t) = point + t * direction
template <typename T>
struct Line {
  Vec3<T> point;
  Vec3<T> direction;
  Line( Vec3<T>& x1, Vec3<T>& x2 ) : point( x1 ), direction( x2 - x1 ) {}
};

/*Input - two ints (x and y) - usually the mouse coordinates 
 *Returns six values in a vector - the first three xf,yf,zf are the coordinates returned by gluUnProject for the point closest to the viewpoint, the last three xb,yb,zb are the coordinates farthest away to the viewpoint in the model space
*/
Line<double> unProject( int x, int y )
{
  GLdouble modelMatrix[16];
  glGetDoublev(GL_MODELVIEW_MATRIX,modelMatrix);
  GLdouble projMatrix[16];
  glGetDoublev(GL_PROJECTION_MATRIX,projMatrix);
  int viewport[4];
  glGetIntegerv(GL_VIEWPORT,viewport);

  Vec3<double> backPoint, frontPoint;
	
  gluUnProject( x, winH - y, 1.0,
		modelMatrix, projMatrix, viewport,
		&backPoint.x, &backPoint.y, &backPoint.z);

  gluUnProject( x, winH - y, 0.0,
		modelMatrix, projMatrix, viewport,
		&frontPoint.x, &frontPoint.y, &frontPoint.z);

  return Line<double>( frontPoint, backPoint );
}

/* Creates a force matrix that can be used by the solver.
input-
   keynode is the index in the mesh of the node that will experience the force
   adjNodes is a list of the nodes adjacent to the keynode.
   x and y are coordinates used to determined the magnitude and direction of
   the force. They are usually the coordinates of the mouse
*/
void createForceMatrix( const matrix<MY_REAL>& coord,
			const Keynode& keynode,
			int x, int y )
{
  int k = (int) keynode;
  Vec3<MY_REAL> keyP( coord(k,0), coord(k,1), coord(k,2) );
  
  Line<double> projLine = unProject(x,y);

  Vec3<double>& lineP = projLine.point;
  Vec3<double>& lineV = projLine.direction;
  
  Vec3<double> forceV = lineP - keyP;

  // Align the force with the plane of the screen
  forceV -= lineV * (lineV.dot(forceV)/lineV.magSq());

  list< pair<int,float> >::const_iterator li;
  for( li = keynode.adjNodes.begin(); li != keynode.adjNodes.end(); ++li ) {
    int n = li->first;
    float scale = li->second;
    forceMatrix(n,0) = scale * forceV.x;
    forceMatrix(n,1) = scale * forceV.y;
    forceMatrix(n,2) = scale * forceV.z;
  }
}


/* Finds the closest node to the point clicked on by the user.
input-
   coord is the matrix containing the mesh coordinates.
   nodeList contains the list of nodes that you want to look through
       T is a type that is castable to an int
   x,y are the coordinates of the mouse
   dist_cutoff 
output-
   The index into nodeList which is closest to the unprojected (x,y)
   NULL_NODE if the closet is not less than the dist_cutoff
*/
template <typename T>
int findNodeIndex( const matrix<MY_REAL>& coord,
		   const vector<T>& nodeList,
		   int x, int y,
		   double dist_cutoff = 1e10 )
{
  dist_cutoff *= dist_cutoff;

  Line<double> projLine = unProject(x,y);

  Vec3<double>& lineP = projLine.point;
  Vec3<double>& lineV = projLine.direction;
  double lineVmagSq = lineV.magSq();
	
  double h_minSq = dist_cutoff;
  int node_min = NULL_NODE;

  // Loop through nodes
  for( int i = 0; i < nodeList.size(); ++i ) {
    int n = (int) nodeList[i];
    Vec3<double> nX( coord(n,0), coord(n,1), coord(n,2) );
	
    // Get the shortest vector between the point and line
    nX -= lineP;
    nX -= lineV * (lineV.dot(nX)/lineVmagSq);

    // Find shortest
    double hSq = nX.magSq();
    if( hSq < h_minSq ) {
      h_minSq = hSq;
      node_min = i;
    }
  }
  
  return (h_minSq < dist_cutoff  ?  node_min  :  NULL_NODE);
}


/*****************************/
/* Global Callback Functions */
/*****************************/

int update_count = 0;

void display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Construct the View Matrix
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  camera.render();

  // Render from the VBO
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexPointer(nDim, GL_REAL, 0, BUF_OFF(0));
  //glColorPointer(3, GL_FLOAT, sizeof(NodalData), BUF_OFF(3*sizeof(float)));
  glEnableClientState(GL_VERTEX_ARRAY);
  //glEnableClientState(GL_COLOR_ARRAY);	
  glColor3f(1.0, 1.0, 1.0); //Hand Color
  //glDrawArrays(GL_POINTS, 0, nNodes);
  
  // Render from the IBO
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glDrawElements(GL_TRIANGLES, ibo_size, GL_UNSIGNED_INT, BUF_OFF(0));
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glDisableClientState(GL_VERTEX_ARRAY);
	
  
  if( keynodeList.size() > 0 ) {
    const matrix<MY_REAL>& coord = solver->getCoord();
    
    // Draw keynodes (Optimize?)
    for( int n = 0; n < keynodeList.size(); ++n ) {
      //keynodeList[n].drawKeynode(1.0,0.0,0.0);

      int gn = (int) keynodeList[n];
      if( n != active_keynode ) {
	Keynode::drawKeynode(coord(gn,0), coord(gn,1), coord(gn,2),
			     1.0, 0.0, 0.0);
      } else {
	// Draw the active keynode in different color
	Keynode::drawKeynode(coord(gn,0), coord(gn,1), coord(gn,2),
			     0.0, 0.0, 1.0);
	
	cout << "Update Count: " << ++update_count << endl;
	// Update the force matrix
	createForceMatrix( coord, keynodeList[active_keynode],
			   mouse_old_x, mouse_old_y );
	solver->setForce( forceMatrix );
      }
    }


  }

  // Update the Buffer
  glutSwapBuffers();

  // Save the Frame
  if( record_key == 1 )
    saveFrame();
}

int iter = 0;

void idle()
{
  StopWatch mytimer; mytimer.start();
  
  cudaGLMapBufferObject((void**)&d_vbo, vbo);
  solver->increment();
  solver->updateVBO();
  cudaGLUnmapBufferObject(vbo);
  
  glutPostRedisplay();

  /*
  // If there's an active keynode
  if( active_keynode != NULL_NODE ) {
    // We need to update the force matrix
    createForceMatrix( solver->getCoord(), keynodeList[active_keynode],
		       mouse_old_x, mouse_old_y );
    solver->setForce( forceMatrix );
  }
  */

  double frame_time = mytimer.stop();
  INCR_TOTAL(Frame,frame_time);
  cout << "Frame: " << ++iter << "    " << 1/frame_time << "fps    " << frame_time << "sec" << endl;
  
  COUT_TOTAL(NR);
  COUT_TOTAL(AssemblyKF);
  COUT_TOTAL(AssemblyF);
  COUT_TOTAL(CG);
  COUT_TOTAL(MVM);
  COUT_TOTAL(Transfer);
  COUT_TOTAL(Frame);
}



void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
  //cerr << "Keyboard: " << key << ", " << (int) key << endl;
	
  switch( key ) {
  case 'p':     // Pause
    idle_on_key = !idle_on_key;
    if( idle_on_key == 0 ) { glutIdleFunc(NULL); }
    else                   { glutIdleFunc(idle); }
    break;
  case 27:      // Escape (esc) key
    glBindBuffer(1, vbo);  glDeleteBuffers(1, &vbo);
    glBindBuffer(1, ibo);  glDeleteBuffers(1, &ibo);
    exit(0);
  case 'r':
    cudaGLMapBufferObject((void**)&d_vbo, vbo);
    solver->reset();
    cudaGLUnmapBufferObject(vbo);
    keyboard('c',0,0);
    image_num = 1;
    break;
  case 'R':     // Recording
    record_key = !record_key;
    break;
  case 'c':     // Center View
    const matrix<MY_REAL>& coord = solver->getCoord();
    Vec3<double> model_center(0,0,0);
    for( int n = 0; n < nNodes; ++n ) {
      model_center.x += coord(n,0); 
      model_center.y += coord(n,1); 
      model_center.z += coord(n,2);
    }
    model_center /= nNodes;

    camera.setViewPoint( model_center );
    break;
  }
}


int select_count = 0;

void mouse(int button, int state, int x, int y)
{
  //cerr << "Mouse: " << button << ", " << state << endl;

  mouse_old_x = x;
  mouse_old_y = y;

  // If a button is pressed
  if( state == GLUT_DOWN ) {

    // Add this button to the mouse state
    mouse_state |= 1 << button;

    // Get the Modifiers
    int mod = glutGetModifiers();

    // If shift is down
    if( mod == GLUT_ACTIVE_SHIFT ) {
      // Create a keynode
      const matrix<MY_REAL>& coord = solver->getCoord();
      int new_keynode = findNodeIndex(coord,surfaceNodes,x,y);
      if( new_keynode != NULL_NODE )
	keynodeList.push_back( Keynode(surfaceNodes[new_keynode], coord) );
      glutPostRedisplay();
      return;
    }

    // If control is down
    if( mod == GLUT_ACTIVE_CTRL ) {
      // Delete a keynode
      const matrix<MY_REAL>& coord = solver->getCoord();
      int old_keynode = findNodeIndex(coord,keynodeList,x,y,Keynode::radius);
      if( old_keynode != NULL_NODE )
	keynodeList.erase( keynodeList.begin() + old_keynode );
      glutPostRedisplay();
      return;
    }
    
    //cout << "Select Transfer: " << ++select_count << endl;
    // Figure out if we are seleting a keynode
    const matrix<MY_REAL>& coord = solver->getCoord();
    active_keynode = findNodeIndex(coord,keynodeList,x,y,Keynode::radius);
    
  } else if( state == GLUT_UP ) {

    // Remove this button from the mouse state
    mouse_state &= ~(1 << button);

    // If a keynode was active
    if( active_keynode != NULL_NODE ) {
      // Set the force to zero
      forceMatrix.zero();
      solver->setForce(forceMatrix);
      // Deactivate the keynode
      active_keynode = NULL_NODE;
    }

  }
}


void motion(int x, int y)
{
  int dx = x - mouse_old_x;
  int dy = y - mouse_old_y;
  mouse_old_x = x;
  mouse_old_y = y;

  // If there is an active node, then we don't use mouse for visualization
  if( active_keynode != NULL_NODE )
    return;
  
  if( mouse_state == MOUSE_STATE_MIDDLE || mouse_state == MOUSE_STATE_BOTH ) {
    // Pan
    Vec3<float> dv(-0.004*dx, 0.004*dy, 0.0);
    camera.pan( dv );
  } else if( mouse_state == MOUSE_STATE_LEFT ) {
    // Rotate
    camera.rotateX( 0.005*dy );
    camera.rotateY( -0.005*dx );
  } else if( mouse_state == MOUSE_STATE_RIGHT ) {
    // Zoom
    float dz = (1 - 0.01*dy);
    camera.zoom( dz );
  }
  
  // Redisplay
  glutPostRedisplay();
}

void reshape(int width, int height) 
{
  winW = width; winH = height;
	
  // Set Viewport
  glViewport(0, 0, winW, winH);
  // Reset Projection Matrix
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, winW/float(winH), 0.1, 1000.0);

  // Redisplay
  glutPostRedisplay();
}

void OpenGLViewer( Solver<MY_REAL>* solver_ )
{
  cout << "Starting Visualization..." << endl;

  solver = solver_;
  nDim   = solver->getMesh().nDim();
  nNodes = solver->getMesh().nNodes();

  // Create GL context
  int argc = 0;
  glutInit(&argc, NULL);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(winW, winH);
  glutCreateWindow("CUDA Nonlinear FEM - Cris Cecka");
      
  // Initialize necessary OpenGL extensions
  glewInit();
  if( glewInit() != GLEW_OK ) {
    fprintf(stderr, "GLEW Initialization Failure");
    fflush(stderr);
    return;
  }
  if( !glewIsSupported("GL_VERSION_2_0 " "GL_ARB_pixel_buffer_object") ) {
    fprintf(stderr, "ERROR: Necessary OpenGL extensions missing.");
    fflush(stderr);
    return;
  }
  
  // Create VBO
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, nNodes*nDim*sizeof(MY_REAL), 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Create CUDA access to vbo
  cudaGLRegisterBufferObject(vbo);
  cudaGLMapBufferObject((void**)&d_vbo, vbo);
  solver->initVBO( d_vbo );
  cudaGLUnmapBufferObject(vbo);

  // Allocate Force matrix
  forceMatrix = matrix<MY_REAL>(nNodes,nDim);

  // Initialize IBO
  vector<unsigned int> h_ibo;
  Mesh<MY_REAL>& mesh = solver->getMesh();
  const matrix<int>& IEN = mesh.getIEN();
  const int nNPE = mesh.nNodesPerElem();
  const int nFPE = mesh.nFacesPerElem();
  const int nNPF = mesh.nNodesPerFace();
  // Can only do tetrahedral meshes for now
  assert( nNPE == 4 && nFPE == 4 && nNPF == 3 );
  const int fIndex[4][3] = {{0,1,2},{0,1,3},{0,2,3},{1,2,3}};

  /*
  // Get the whole mesh (4 triangles per tet) //
  h_ibo.resize(mesh.nElems()*nFPE*nNPF);
  // For each elements
  for( int e = 0; e < mesh.nElems(); ++e ) {
  for( int face = 0; face < nFPE; ++face ) {
  for( int a = 0; a < nNPF; ++a ) {
  h_ibo[e*nFPE*nNPF + face*nNPF + a]  = mesh.IEN(e,fIndex[face][a]);
  }
  }
  }
  */

  // Get the surface mesh (Keep triangles on surface)//
  // For each element
  for( int e = 0; e < mesh.nElems(); ++e ) {
    // If this element has neighbors on all faces, it can't have a surface face
    if( mesh.dxadj[e+1] - mesh.dxadj[e] == nFPE ) continue;
    
    // For each node of e, count the number of face-adjacent elements
    map<int,int> nodeTally;
    for( int a = 0; a < nNPE; ++a )
      nodeTally[ IEN(e,a) ] = 0;

    // For each neighbor element
    for( int e2Ptr = mesh.dxadj[e]; e2Ptr < mesh.dxadj[e+1]; ++e2Ptr ) {  
      int e2 = mesh.dadjncy[e2Ptr];

      // For each node of e2
      for( int a2 = 0; a2 < nNPE; ++a2 ) {
	int n2 = IEN(e2,a2);

	// If this node of e2 is also a node of e (is a node we're tallying)
	map<int,int>::iterator mi = nodeTally.find( n2 );
	if( mi != nodeTally.end() ) {
	  // Increment the node tally
	  ++nodeTally[ n2 ];
	}
      }
    }

    // For Tetrahedron...
    // An interior face has a tally sum that is odd
    // A surface face has a tally sum that is even
    for( int face = 0; face < nFPE; ++face ) {
      // Get the node tally sum for this face
      int sum = 0;
      for( int a = 0; a < nNPF; ++a ) 
	sum += nodeTally[ IEN(e,fIndex[face][a]) ];

      // If the node tally sum is even, add this face
      if( ISEVEN(sum) ) {
	for( int a = 0; a < nNPF; ++a ) 
	  h_ibo.push_back( IEN(e,fIndex[face][a]) );
      }
    }
    
  }
	
  // Save the ibo size
  ibo_size = h_ibo.size();
  
  // Create IBO
  glGenBuffers(1, &ibo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, ibo_size*sizeof(h_ibo[0]), &h_ibo[0], 
	       GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	


  // Save a vector of the surface nodes
  surfaceNodes = h_ibo;
  sort(surfaceNodes.begin(),surfaceNodes.end());
  surfaceNodes.erase( unique(surfaceNodes.begin(),surfaceNodes.end()), 
		      surfaceNodes.end() );

  // Background Color
  glClearColor(0.0, 0.0, 0.0, 1.0);  

  // OpenGL Fog for better depth perception
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_POINT_SMOOTH);
  glPointSize(2.0);
  glPointParameterf(GL_POINT_SIZE_MIN,0.0f);
  glPointParameterf(GL_POINT_SIZE_MAX,100.0f);
  float pointparams[3] = {0,0,1};
  glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, pointparams);
  glEnable(GL_FOG);
  glFogi(GL_FOG_MODE, GL_EXP);
  glFogf(GL_FOG_DENSITY,0.3);
	
  // Register callbacks
  glutReshapeFunc(reshape);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutIdleFunc(idle);

  // Initialize Camera View
  camera.zoom(2);
  // Center the View
  keyboard('c',0,0);
  
  // Start the OpenGL Loop
  glutPostRedisplay();
  glutMainLoop();
}


#endif
