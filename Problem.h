#ifndef PROBLEM_H
#define PROBLEM_H

#include "General.h"

#include "Mesh.h"

#include "MyMatrix/Matrix.h"
#include "MyMatrix/MatrixSym.h"
#include "MyMatrix/MatrixDiag.h"
#include "MyMatrix/Vector.h"

#define __KF__ 0
#define __F__  1


/* Master class which constructs an FEM problem to solve
 */

template <typename T>
class Problem
{
  // Mesh Related Data
  Mesh<T>& mesh;
  
 public:
  
  // All data defined by node
  struct NodalData {
    T x, y, z;      // Nodal coordinates
    T fx, fy, fz;   // Nodal forces
  };
  // All data defined by element
  struct SuppData {
    T Jinv[6];      // Element-wise upper triangular reference deformation
  };
  // Local stiffness matrix and forcing vector of an element
  struct ElemData {
    T k_e[78];      // Element stiffness (12x12 symmetric)
    T f_e[12];      // Element forcing
  };
  
  // Problem Related Data
  int nDOF;               // Degrees of freedom per node
  matrix<T> Jinv;         // Jinv (upper 3x3) matrices: Elem # to UMatrix
  
  matrix<T> coord;        // Nodal coordinates: GNode #, Dim to Coord
  matrix<T> p;            // Nodal momenta
  matrix<T> force;        // Array of nodal forces

  // Boundary Condition Data (Not Used)
  matrix<char> BCIndex;   // Array of essential BC indices
  matrix<T> BCVal;        // Array of essential BC values
  matrix<T> h;            // Array of natural BC values
  
 Problem( Mesh<T>& mesh_ )
   : mesh( mesh_ ), nDOF(3),
    Jinv(mesh.nElems(), 6),
    coord( mesh.getCoord() ),
    p(mesh.nNodes(), nDOF), 
    force(mesh.nNodes(), nDOF),
    BCIndex(mesh.nNodes(), nDOF), 
    BCVal(mesh.nNodes(), nDOF),
    h(mesh.nNodes(), nDOF)
      {
	const matrix<int>& IEN = mesh.getIEN();

	matrix<double> Dm( 3, 3 );
	matrix<double> Q( 3, 3 );
	matrix<double> DmR( 3, 3 );

	// Precompute all the upper triangular Jinv matrices on the mesh
	// Need to do this accurately!
	for( int e = 0; e < mesh.nElems(); ++e ) {
	  int n0 = IEN(e,0), n1 = IEN(e,1), n2 = IEN(e,2), n3 = IEN(e,3);
	  
	  // Get the reference node coordinates
	  double rx1 = coord(n0,0), ry1 = coord(n0,1), rz1 = coord(n0,2);
	  double rx2 = coord(n1,0), ry2 = coord(n1,1), rz2 = coord(n1,2);
	  double rx3 = coord(n2,0), ry3 = coord(n2,1), rz3 = coord(n2,2);
	  double rx4 = coord(n3,0), ry4 = coord(n3,1), rz4 = coord(n3,2);
	  
	  // Compute Dm
	  Dm(0,0) = rx1 - rx4; Dm(0,1) = rx2 - rx4; Dm(0,2) = rx3 - rx4;
	  Dm(1,0) = ry1 - ry4; Dm(1,1) = ry2 - ry4; Dm(1,2) = ry3 - ry4;
	  Dm(2,0) = rz1 - rz4; Dm(2,1) = rz2 - rz4; Dm(2,2) = rz3 - rz4;
	  
	  // Need to QR the Dm!!  Use Gram–Schmidt...
	  for( int j = 0; j < 3; ++j ) {
	    Q(0,j) = Dm(0,j); 
	    Q(1,j) = Dm(1,j); 
	    Q(2,j) = Dm(2,j);
	    for( int k = 0; k < j; ++k ) {
	      double dotP = Q(0,k)*Dm(0,j) + Q(1,k)*Dm(1,j) + Q(2,k)*Dm(2,j);
	      Q(0,j) -= dotP * Q(0,k);
	      Q(1,j) -= dotP * Q(1,k);
	      Q(2,j) -= dotP * Q(2,k);
	    }
	    double normQ = sqrt(Q(0,j)*Q(0,j) + Q(1,j)*Q(1,j) + Q(2,j)*Q(2,j));
	    Q(0,j) /= normQ;
	    Q(1,j) /= normQ;
	    Q(2,j) /= normQ;
	  }
	  
	  // Compute DmR = Q^T Dm
	  DmR(0,0) = Q(0,0)*Dm(0,0) + Q(1,0)*Dm(1,0) + Q(2,0)*Dm(2,0);
	  DmR(0,1) = Q(0,0)*Dm(0,1) + Q(1,0)*Dm(1,1) + Q(2,0)*Dm(2,1);
	  DmR(0,2) = Q(0,0)*Dm(0,2) + Q(1,0)*Dm(1,2) + Q(2,0)*Dm(2,2);

	  DmR(1,0) = Q(0,1)*Dm(0,0) + Q(1,1)*Dm(1,0) + Q(2,1)*Dm(2,0);
	  DmR(1,1) = Q(0,1)*Dm(0,1) + Q(1,1)*Dm(1,1) + Q(2,1)*Dm(2,1);
	  DmR(1,2) = Q(0,1)*Dm(0,2) + Q(1,1)*Dm(1,2) + Q(2,1)*Dm(2,2);

	  DmR(2,0) = Q(0,2)*Dm(0,0) + Q(1,2)*Dm(1,0) + Q(2,2)*Dm(2,0);
	  DmR(2,1) = Q(0,2)*Dm(0,1) + Q(1,2)*Dm(1,1) + Q(2,2)*Dm(2,1);
	  DmR(2,2) = Q(0,2)*Dm(0,2) + Q(1,2)*Dm(1,2) + Q(2,2)*Dm(2,2);
	  
	  // Make sure this is actually upper triangular
	  assert( DmR(1,0) < 1e-15 && DmR(2,0) < 1e-15 && DmR(2,1) < 1e-15 );

	  // Undo any inversion in the QR
	  double Qdet =  Q(0,0)*(Q(1,1)*Q(2,2)-Q(1,2)*Q(2,1))
	                +Q(0,1)*(Q(1,2)*Q(2,0)-Q(1,0)*Q(2,2))
	                +Q(0,2)*(Q(1,0)*Q(2,1)-Q(1,1)*Q(2,0));
	  if( Qdet < 0 )
	    DmR(2,2) = -DmR(2,2);
	  
	  // Compute Jinv = DmR^-1 and keep upper triangular part
	  double invdet = 1.0 / (DmR(0,0)*DmR(1,1)*DmR(2,2));
	  Jinv(e,0) = 1.0 / DmR(0,0);                               // DmRInv00
	  Jinv(e,1) = (DmR(0,2)*DmR(2,1)-DmR(0,1)*DmR(2,2))*invdet; // DmRInv01
	  Jinv(e,2) = (DmR(0,1)*DmR(1,2)-DmR(0,2)*DmR(1,1))*invdet; // DmRInv02
	  Jinv(e,3) = 1.0 / DmR(1,1);                               // DmRInv11
	  Jinv(e,4) = (DmR(0,2)*DmR(1,0)-DmR(0,0)*DmR(1,2))*invdet; // DmRInv12
	  Jinv(e,5) = 1.0 / DmR(2,2);                               // DmRInv22
	}
      }
  
  inline Mesh<T>& getMesh()
  {
    return mesh;
  }

  inline matrix<T>& getCoord()
  {
    return coord;
  }

  inline matrix<T>& getForce()
  {
    return force;
  }

  inline matrix<T>& getMomentum()
  {
    return p;
  }

  inline int nNDOF() const
  {
    return nDOF;
  }

  inline int nEDOF() const
  {
    return nNDOF() * mesh.nNodesPerElem();
  }

  inline static int nodalDataSize()
  {
    return sizeof(NodalData);
  }

  inline static int suppDataSize()
  {
    return sizeof(SuppData);
  }

  inline static int elemDataSize()
  {
    return sizeof(ElemData);
  }

  inline vector<T> getSuppData(int e) const
  {
    vector<T> suppData(6);
    suppData[0] = Jinv(e,0);
    suppData[1] = Jinv(e,1);
    suppData[2] = Jinv(e,2);
    suppData[3] = Jinv(e,3);
    suppData[4] = Jinv(e,4);
    suppData[5] = Jinv(e,5);
    return suppData;
  }


  /**********************************************/
  /************* Element Kernels ****************/
  /**********************************************/


  inline void tetrahedralMass( int e, dmatrix<T>& m_e )
  {
    T V = (ORIENT/((T)6.0))/(Jinv(e,0)*Jinv(e,3)*Jinv(e,5));
    assert( V > 0 );

    int N = m_e.size();
    for( int i = 0; i < N; ++i ) {
      m_e[i] = MASS * V / ((T)4);
    }
  }

  template <int TYPE, int ESTRIDE>
  static __inline__ __device__ 
  void Tetrahedral( const T x1,  const T y1,  const T z1,
		    const T bx1, const T by1, const T bz1,
		    const T x2,  const T y2,  const T z2,
		    const T bx2, const T by2, const T bz2,
		    const T x3,  const T y3,  const T z3,
		    const T bx3, const T by3, const T bz3,
		    const T x4,  const T y4,  const T z4,	  
		    const T bx4, const T by4, const T bz4,
		    const T Jinv11, const T Jinv12, const T Jinv13,
		    const T Jinv22, const T Jinv23,
		    const T Jinv33,
		    T* E )
    {
#if TYPE == __KF__
#include "MyAssembly/elemRepo/elastKF_GPU.ker"
#elif TYPE == __F__
#include "MyAssembly/elemRepo/elastF_GPU.ker"
#else
      ERROR!
#endif
    }
  

  void tetrahedralElasticity( int e, matrix<T>& C, matrix<T>& f,
			      smatrix<T>& k_e, vector<T>& f_e )
  {
    const matrix<int>& IEN = mesh.getIEN();
    int ien[4] = {IEN(e,0), IEN(e,1), IEN(e,2), IEN(e,3)};

    T x1 = C(ien[0],0), y1 = C(ien[0],1), z1 = C(ien[0],2);
    T x2 = C(ien[1],0), y2 = C(ien[1],1), z2 = C(ien[1],2);
    T x3 = C(ien[2],0), y3 = C(ien[2],1), z3 = C(ien[2],2);
    T x4 = C(ien[3],0), y4 = C(ien[3],1), z4 = C(ien[3],2);
    
    // Get the Material Jacobian
    T Jinv11 = Jinv(e,0), Jinv12 = Jinv(e,1), Jinv13 = Jinv(e,2);
    T Jinv22 = Jinv(e,3), Jinv23 = Jinv(e,4);
    T Jinv33 = Jinv(e,5);

    // Get the body forces
    T bx1 = f(ien[0],0), by1 = f(ien[0],1), bz1 = f(ien[0],2);
    T bx2 = f(ien[1],0), by2 = f(ien[1],1), bz2 = f(ien[1],2);
    T bx3 = f(ien[2],0), by3 = f(ien[2],1), bz3 = f(ien[2],2);
    T bx4 = f(ien[3],0), by4 = f(ien[3],1), bz4 = f(ien[3],2);

    #include "MyAssembly/elemRepo/elastKF_CPU.ker"
  }


  void tetrahedralForce( int e, matrix<T>& C, matrix<T>& f, 
			 vector<T>& f_e )
  {
    const matrix<int>& IEN = mesh.getIEN();
    int ien[4] = {IEN(e,0), IEN(e,1), IEN(e,2), IEN(e,3)};

    T x1 = C(ien[0],0), y1 = C(ien[0],1), z1 = C(ien[0],2);
    T x2 = C(ien[1],0), y2 = C(ien[1],1), z2 = C(ien[1],2);
    T x3 = C(ien[2],0), y3 = C(ien[2],1), z3 = C(ien[2],2);
    T x4 = C(ien[3],0), y4 = C(ien[3],1), z4 = C(ien[3],2);
    
    // Get the Material Jacobian
    T Jinv11 = Jinv(e,0), Jinv12 = Jinv(e,1), Jinv13 = Jinv(e,2);
    T Jinv22 = Jinv(e,3), Jinv23 = Jinv(e,4);
    T Jinv33 = Jinv(e,5);

    // Get the body forces
    T bx1 = f(ien[0],0), by1 = f(ien[0],1), bz1 = f(ien[0],2);
    T bx2 = f(ien[1],0), by2 = f(ien[1],1), bz2 = f(ien[1],2);
    T bx3 = f(ien[2],0), by3 = f(ien[2],1), bz3 = f(ien[2],2);
    T bx4 = f(ien[3],0), by4 = f(ien[3],1), bz4 = f(ien[3],2);

    #include "MyAssembly/elemRepo/elastF_CPU.ker"
  }
};

#endif
