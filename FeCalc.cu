#ifndef FECALC_CU
#define FECALC_CU


#include "General.h"
#include "General.cu"

#include "MyMatrix/COO_Matrix.h"
#include "MyMatrix/CSR_Matrix.h"
#include "MyMatrix/HYB_Matrix.h"
#include "MyMatrix/DCOO_Matrix.h"
#include "MyMatrix/DCSR_Matrix.h"
#include "MyMatrix/DHYB_Matrix.h"
#include "MyMatrix/DHYBC_Matrix.h"


#include "Mesh.h"
#include "Problem.h"

#include "MyAssembly/AssemblySerial.h"
#include "MyAssembly/AssemblySerialOpt.h"
#include "MyAssembly/AssemblyGlobalNZ.h"
#include "MyAssembly/AssemblySharedNZ.h"

#include "MyCG/CG_CPU.h"
#include "MyCG/CG_GPU.h"

#include "Parser.h"

#include "MySolver/SolverCPU.h"
#include "MySolver/SolverGPU.h"

#include "MyOpenGL/OpenGLViewer.h"


#if 1

// Main runner
int main( int argc, char* argv[] )
{
  (void) argc;
  (void) argv; // quiet compiler

  cudaInit();

  matrix<MY_REAL> coord;
  matrix<int> IEN;

  int nDim = 3;
  int nNPE = 4;

  //string fileOFF = "data/test.off";
  //string fileOFF = "data/test2.off";
  //string fileOFF = "data/hand28796v.off";
  //parseOFF( fileOFF.c_str(), coord, nDim, IEN, nNPE );

  //string fileDAT = "data/Sphere_63.dat";
  //string fileDAT = "data/Sphere_305.dat";
  //string fileDAT = "data/Sphere_407.dat";
  //string fileDAT = "data/Sphere_407.dat";
  //string fileDAT = "data/Sphere_1701.dat";
  //string fileDAT = "data/Sphere_25351.dat";
  string fileDAT = "data/Torus_20k.dat";
  //string fileDAT = "data/Torus_30k.dat";
  //string fileDAT = "data/Torus_37k.dat";
  //string fileDAT = "data/Torus_45k.dat";
  //string fileDAT = "data/Torus_55k.dat";
  parseDAT( fileDAT.c_str(), coord, nDim, IEN, nNPE );

  cout << "Points: " << coord.nRows() << endl;
  cout << "Elements: " << IEN.nRows() << endl;

  cout << "Defining Mesh..." << endl;
  Mesh<MY_REAL> mesh( coord, IEN );

  cout << "Defining Problem..." << endl;
  Problem<MY_REAL> problem( mesh );

  cout << "Defining Matrix..." << endl;
  matrix_dhyb<MY_REAL> FEM_Matrix;

  cout << "Defining Assembler..." << endl;
  AssemblySharedNZ<MY_REAL> assembler( problem, FEM_Matrix );

  cout << "NZs: " << FEM_Matrix.nNZ() << endl;

  cout << "Defining MVM..." << endl;
  DHYB_MVM_GPU<MY_REAL> mvm( FEM_Matrix );

  cout << "Defining CGer..." << endl;
  DCG_GPU<MY_REAL> cger( mvm );

  cout << "Defining Solver..." << endl;
  Solver<MY_REAL>* solver = new NR_GPU<MY_REAL>( &assembler, &cger );

  // Solver that does nothing (testing OpenGL)
  //Solver<MY_REAL>* solver = new Solver<MY_REAL>( problem );

  OpenGLViewer( solver );

  return 0;
}

#endif





#if 0

////////////////////////////////////////////
////////// Test Assembly and MVM ///////////
////////////////////////////////////////////

template <class T1, class T2>
inline double checkData( const vector<T1>& val,
			 const vector<T2>& exact,
			 bool out = false )
{
  //cout << val.size() << "    " << exact.size() << endl;
  assert( val.size() == exact.size() );

  int N = val.size();
  double error = 0;
  double maxE = 0;
  double errorR = 0;
  double maxR = 0;
  double errorL2 = 0;
  double normL2 = 0;

  for( int k = 0; k < N; ++k ) {
    double e = abs( val[k] - exact[k] );
    double er = abs( val[k] - exact[k] ) / abs( exact[k] );
    double el2 = ( val[k] - exact[k] ) * ( val[k] - exact[k] );
    double rl2 = exact[k] * exact[k];

    if( out )
      cerr << k << ": " << val[k] << "\t" << exact[k] << "\t"
	   << e << "\t" << er << endl;

    maxE = max( maxE, e );
    error += e;
    errorR += er;
    maxR = max( maxR, er );
    errorL2 += el2;
    normL2 += rl2;
  }

  cerr << "AveError: " << left << setw(13) << maxE
       << "AveRelE: "  << left << setw(12) << errorR/N
       << left << setw(12) << maxR
       << "L2RelError: " << sqrt(errorL2/normL2) << endl;
  return error;
}



template <template <typename> class MATRIX,
	  template <typename,template <typename> class> class ASSEMBLER,
	  template <typename> class MVMTYPE,
	  typename T,
	  typename TREF>
inline void testType( matrix_sparse<TREF>& Kexact,
		      vector<TREF>& Fexact,
		      vector<TREF>& Yexact,
		      Problem<T>& problem )
{
  cout << "Constructing " << ASSEMBLER<T,MATRIX>::name()
       << " " << MATRIX<T>::name() << "..." << endl;

  ASSEMBLER<T,MATRIX> assembler( problem );

  vector<T> forces( assembler.nEquation() );
  assembler.arrangeData( assembler.getProblem().force, forces );
  vector<T> coords( assembler.nEquation() );
  assembler.arrangeData( assembler.getProblem().coord, coords );

  cerr << "Checking Assembled " << MATRIX<T>::name() << " + F" << endl;
  assembler.assembleKF( coords, forces );
  checkData( assembler.getK().getCSRA(), Kexact.getCSRA() );
  checkData( assembler.getF(), Fexact );

  cerr << "Checking Assembled F" << endl;
  assembler.assembleF( coords, forces );
  checkData( assembler.getF(), Fexact );

  cerr << "Checking " << MVMTYPE<T>::name() << endl;
  MVMTYPE<T> mvm( assembler.getK() );

  //#if T == TREF
  //mvm.prod_cpu( (vector<T>&) assembler.getK(), Fexact );
  //checkData( mvm.getY(), Yexact );
  //#endif

  mvm.prod_cpu( (vector<T>&) assembler.getK(), assembler.getF() );
  checkData( mvm.getY(), Yexact );

  cout << endl;
}


// Validator
int main( int argc, char* argv[] )
{
  cout << endl;

  // Test case with known values
  // To generalize, read from input files
  int nDim = 3;
  int nNPE = 4;

  // Constructing a double-precision reference solution
  cout << "Constructing Reference Problem..." << endl;

#define REAL_REF double

  matrix<REAL_REF> refcoord;
  matrix<int> refIEN;

  //string fileOFF = "data/test.off";
  //string fileOFF = "data/test2.off";
  string fileOFF = "data/hand28796v.off";
  parseOFF( fileOFF.c_str(), refcoord, nDim, refIEN, nNPE );

  //string fileDAT = "data/Sphere_63.dat";
  //string fileDAT = "data/Sphere_305.dat";
  //string fileDAT = "data/Sphere_407.dat";
  //string fileDAT = "data/Sphere_407.dat";
  //string fileDAT = "data/Sphere_1701.dat";
  //string fileDAT = "data/Sphere_3611.dat";
  //string fileDAT = "data/Sphere_25431.dat";
  //parseDAT( fileDAT.c_str(), refcoord, nDim, refIEN, nNPE );

  //cout << coord << endl << endl;
  //cout << IEN << endl << endl;

  cout << "Points: " << refcoord.nRows() << endl;
  cout << "Elements: " << refIEN.nRows() << endl;

  cout << "Meshing..." << endl;
  Mesh<REAL_REF> refmesh( refcoord, refIEN );

  cout << "Defining Problem..." << endl;
  Problem<REAL_REF> refproblem( refmesh );

  // Expand!
  for( int n = 0; n < refproblem.coord.size(); ++n ) {
    refproblem.coord[n] *= 12.0/8.0;
  }

  cout << "Defining Serial Assembler..." << endl;
  AssemblySerial<REAL_REF,matrix_coo> assembler( refproblem );

  cout << "Computing Reference Solutions..." << endl;
  vector<REAL_REF> forces( assembler.nEquation() );
  assembler.arrangeData( assembler.getProblem().force, forces );
  vector<REAL_REF> coords( assembler.nEquation() );
  assembler.arrangeData( assembler.getProblem().coord, coords );

  // Compute the assembly
  assembler.assembleKF( coords, forces );
  matrix_sparse<REAL_REF>& Kcoo = assembler.getK();
  vector<REAL_REF>& Fcoo = assembler.getF();

  // Compute a matrix-vector product
  COO_MVM_CPU<REAL_REF> coo_mvm( assembler.getK() );
  coo_mvm.prod_cpu( (vector<REAL_REF>&) Kcoo, Fcoo );
  vector<REAL_REF>& Ycoo = coo_mvm.getY();

  cout << endl;


  cout << "Constructing Test Problem..." << endl;

#define REAL_TEST float

  matrix<REAL_TEST> coord;
  matrix<int> IEN;

  parseOFF( fileOFF.c_str(), coord, nDim, IEN, nNPE );
  //parseDAT( fileDAT.c_str(), coord, nDim, IEN, nNPE );

  //cout << coord << endl << endl;
  //cout << IEN << endl << endl;

  cout << "Points: " << coord.nRows() << endl;
  cout << "Elements: " << IEN.nRows() << endl;

  cout << "Meshing..." << endl;
  Mesh<REAL_TEST> mesh( coord, IEN );

  cout << "Defining Problem..." << endl;
  Problem<REAL_TEST> problem( mesh );

  // Expand!
  for( int n = 0; n < problem.coord.size(); ++n ) {
    problem.coord[n] *= 12.0/8.0;
  }

  cout << endl;

  testType<matrix_coo,AssemblySerial,COO_MVM_CPU>
      ( Kcoo, Fcoo, Ycoo, problem );

  testType<matrix_coo,AssemblySerialOpt,COO_MVM_CPU>
      ( Kcoo, Fcoo, Ycoo, problem );

  testType<matrix_csr,AssemblySharedNZ,CSR_MVM_GPU_Vector>
      ( Kcoo, Fcoo, Ycoo, problem );

  testType<matrix_hyb,AssemblySharedNZ,HYB_MVM_GPU>
      ( Kcoo, Fcoo, Ycoo, problem );

  testType<matrix_csr,AssemblyGlobalNZ,CSR_MVM_GPU_Vector>
      ( Kcoo, Fcoo, Ycoo, problem );

  testType<matrix_hyb,AssemblyGlobalNZ,HYB_MVM_GPU>
      ( Kcoo, Fcoo, Ycoo, problem );

  return 0;
}

#endif



#endif
