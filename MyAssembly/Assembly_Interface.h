#ifndef ASSEMBLYENV_H
#define ASSEMBLYENV_H

#include "../General.h"

#include "../Problem.h"

#include "../MyMatrix/Matrix.h"
#include "../MyMatrix/MatrixSym.h"
#include "../MyMatrix/MatrixDiag.h"
#include "../MyMatrix/Vector.h"


template <typename T>
class Assembly_Interface
{
 public:
  Problem<T>& prob;

  // Assembly Related Data
  int nEquations;
  matrix<int> ID;                             // GNode #, NDOF # to GEQ #
  matrix<int> LM;                             // Elem #, EDOF # to GEQ #

  // Empty (diagonal) lumped mass matrix
  dmatrix<T> M_;
  // Empty sparse profile of the stiffness matrix
  matrix_sparse<T> K_;
  // Empty forcing vector
  vector_cpu<T> F_;

 public:

  template <template <typename> class MATRIX>
  Assembly_Interface( Problem<T>& p, MATRIX<T>& FEM_Matrix )
      : prob(p), nEquations(0),
        ID( prob.getMesh().nNodes(), prob.nNDOF() ),
        LM( prob.getMesh().nElems(), prob.nEDOF() )
  {
    /* Compute the global data processing arrays */
    Mesh<T>& mesh = prob.getMesh();
    //matrix<char>& BCIndex = prob->BCIndex;

    int nElems = mesh.nElems();
    int nNPE = mesh.nNodesPerElem();
    //int nDim = mesh.nDim();
    int nNodes = mesh.nNodes();
    int nDoF = prob.nNDOF();
    //int eDoF = prob.nEDOF();

    matrix<char>& BCIndex = prob.BCIndex;
    // Give each NDOF an equation number
    for( int n = 0; n < nNodes; ++n ) {
      for( int d = 0; d < nDoF; ++d ) {
        if( !BCIndex(n,d) ) {
          // Prescribe this DOF an equation number
          ID(n,d) = nEquations;
          ++nEquations;
        } else {
          // this is a prescribed BC, do not assign equation #
          ID(n,d) = -1;
        }
      }
    }

    const matrix<int>& IEN = mesh.getIEN();

    // The connectivity matrix
    list< pair<int,int> > IJList;

    // Determine the equation number by element and local node
    for( int e = 0; e < nElems; ++e ) {
      for( int a = 0; a < nNPE; ++a ) {
        for( int d = 0; d < nDoF; ++d ) {
          int eq1 = ID( IEN(e,a), d );
          int m = a*nDoF + d;
          LM(e,m) = eq1;
          if( eq1 == -1 ) continue;

          IJList.push_back( make_pair(eq1,eq1) );

          for( int m2 = 0; m2 < m; ++m2 ) {
            int eq2 = LM(e,m2);
            if( eq2 == -1 ) continue;

            IJList.push_back( make_pair(eq1,eq2) );
            IJList.push_back( make_pair(eq2,eq1) );
          }
        }
      }
    }

    M_ = dmatrix<T>( nEquations );
    FEM_Matrix = MATRIX<T>( IJList );
    K_ = FEM_Matrix;
    F_ = vector_cpu<T>( nEquations );
  }


  // Destructor
  virtual ~Assembly_Interface() {}

  static string name() { return "Assembly_Interface"; }

  // Accessors
  inline int nEquation() { return nEquations; }
  inline Problem<T>& getProblem() { return prob; }

  // Define the CPU Interface //
  virtual void assembleM_cpu() = 0;
  virtual void assembleKF_cpu(vector_cpu<T>& coord, vector_cpu<T>& force) = 0;
  virtual void assembleF_cpu(vector_cpu<T>& coord, vector_cpu<T>& force) = 0;
  virtual dmatrix<T>&       getM_cpu() = 0;
  virtual matrix_sparse<T>& getK_cpu() = 0;
  virtual vector_cpu<T>&    getF_cpu() = 0;

  // Define the GPU Interface //
  virtual void assembleM_gpu() = 0;
  virtual void assembleKF_gpu(vector_gpu<T>& coord, vector_gpu<T>& force) = 0;
  virtual void assembleF_gpu(vector_gpu<T>& coord, vector_gpu<T>& force) = 0;
  virtual vector_gpu<T>& getM_gpu() = 0;
  virtual vector_gpu<T>& getK_gpu() = 0;
  virtual vector_gpu<T>& getF_gpu() = 0;


  /***** Standard serial helper methods *****/

  inline void arrangeData( matrix<T>& nodedata, vector<T>& x )
  {
    int nNodes = prob.mesh.nNodes();
    int nDoF = prob.nNDOF();
    for( int n = 0; n < nNodes; ++n ) {
      for( int d = 0; d < nDoF; ++d ) {
	int eq = ID(n,d);
	if( eq == -1 ) continue;

	x[eq] = nodedata(n,d);
      }
    }
  }

  inline void arrangeResults( vector<T>& x, matrix<T>& result )
  {
    matrix<char>& BCIndex = prob.BCIndex;
    matrix<T>& BCVal = prob.BCVal;
    int nNodes = prob.mesh.nNodes();
    int nDoF = prob.nNDOF();
    for( int n = 0; n < nNodes; ++n ) {
      for( int d = 0; d < nDoF; ++d ) {
	if( BCIndex(n,d) ) {
	  result(n,d) = BCVal(n,d);
	} else {
	  result(n,d) = x[ID(n,d)];
	}
      }
    }
  }

  inline void assembleStiffness( int e,
				 smatrix<T>& k_e, matrix_sparse<T>& K )
  {
    int eDoF = prob.nEDOF();
    for( int i = 0; i < eDoF; ++i ) {
      int LMi = LM(e,i);
      if( LMi == -1 ) continue;

      // Note Symmetry of k_e!!
      for( int j = i; j < eDoF; ++j ) {
	int LMj = LM(e,j);
	if( LMj == -1 ) continue;

	K(LMi,LMj) += k_e(i,j);
      }
    }
  }

  inline void assemble( int e,
			dmatrix<T>& m_e, dmatrix<T>& M )
  {
    int eDoF = prob.nEDOF();
    for( int i = 0; i < eDoF; ++i ) {
      int LMi = LM(e,i);
      if( LMi == -1 ) continue;

      M[LMi] += m_e[i];
    }
  }

  inline void assemble( int e,
			vector<T>& f_e, vector<T>& F )
  {
    int eDoF = prob.nEDOF();
    for( int i = 0; i < eDoF; ++i ) {
      int LMi = LM(e,i);
      if( LMi == -1 ) continue;

      F[LMi] += f_e[i];
    }
  }

  inline void assemble( int e,
			dmatrix<T>& m_e, dmatrix<T>& M,
			smatrix<T>& k_e, matrix_sparse<T>& K,
			vector<T>& f_e, vector<T>& F )
  {
    int eDoF = prob.nEDOF();
    for( int i = 0; i < eDoF; ++i ) {
      int LMi = LM(e,i);
      if( LMi == -1 ) continue;

      F[LMi] += f_e[i];
      M[LMi] += m_e[i];

      for( int j = 0; j < eDoF; ++j ) {
	int LMj = LM(e,j);
	if( LMj == -1 ) continue;

	K(LMi,LMj) += k_e(i,j);
      }
    }
  }

  inline void assemble( int e,
			smatrix<T>& k_e, matrix_sparse<T>& K,
			vector<T>& f_e, vector<T>& F )
  {
    int eDoF = prob.nEDOF();
    for( int i = 0; i < eDoF; ++i ) {
      int LMi = LM(e,i);
      if( LMi == -1 ) continue;

      F[LMi] += f_e[i];

      // Note Symmetry of k_e!!
      for( int j = 0; j < eDoF; ++j ) {
	int LMj = LM(e,j);
	if( LMj == -1 ) continue;

	K(LMi,LMj) += k_e(i,j);
      }
    }
  }

  inline void addNaturalBC( vector<T>& F )
  {
    matrix<T>& h = prob.h;
    int nNodes = prob.mesh.nNodes();
    int nDoF = prob.nNDOF();
    for( int n = 0; n < nNodes; ++n ) {
      for( int d = 0; d < nDoF; ++d ) {
	int k = ID(n,d);
	if( k == -1 ) continue;

	F[k] += h(n,d);
      }
    }
  }

};



template <typename T>
class AssemblyCPU : public Assembly_Interface<T>
{
  // Staging area for possible transfers
  vector_cpu<T> coord;              // Nodal coords
  vector_cpu<T> force;              // Nodal forces

  // Storage on the GPU for gpu interface
  vector_gpu<T> d_M;
  vector_gpu<T> d_K;
  vector_gpu<T> d_F;

 protected:

  using Assembly_Interface<T>::M_;
  using Assembly_Interface<T>::K_;
  using Assembly_Interface<T>::F_;

 public:

  // Constructor
  template <template <typename> class MATRIX>
  AssemblyCPU( Problem<T>& p, MATRIX<T>& FEM_Matrix )
      : Assembly_Interface<T>( p, FEM_Matrix ),
      coord( p.coord.size() ),
      force( p.force.size() ),
      d_M( M_ ),
      d_K( K_ ),
      d_F( F_ ) {}
  // Destructor
  virtual ~AssemblyCPU() {}

  static string name() { return "AssemblyCPU"; }

  // Define the CPU Interface //
  virtual void assembleM_cpu() = 0;
  virtual void assembleKF_cpu(vector_cpu<T>& coord, vector_cpu<T>& force) = 0;
  virtual void assembleF_cpu(vector_cpu<T>& coord, vector_cpu<T>& force) = 0;
  inline dmatrix<T>&       getM_cpu() { return M_; }
  inline matrix_sparse<T>& getK_cpu() { return K_; }
  inline vector_cpu<T>&    getF_cpu() { return F_; }

  // Implement the GPU Interface in terms of the CPU Interface //
  inline void assembleM_gpu() { return assembleM_cpu(); }
  inline void assembleKF_gpu( vector_gpu<T>& d_coord, vector_gpu<T>& d_force )
  {
    coord = d_coord; force = d_force;
    return assembleKF_cpu( coord, force );
  }
  inline void assembleF_gpu( vector_gpu<T>& d_coord, vector_gpu<T>& d_force )
  {
    coord = d_coord; force = d_force;
    return assembleF_cpu( coord, force );
  }
  inline vector_gpu<T>& getM_gpu() { return d_M = M_; }
  inline vector_gpu<T>& getK_gpu() { return d_K = K_; }
  inline vector_gpu<T>& getF_gpu() { return d_F = F_; }
};


template <typename T>
class AssemblyGPU : public Assembly_Interface<T>
{
  // Staging area for possible transfer
  vector_gpu<T> d_coord;              // Device pointer to nodal coords
  vector_gpu<T> d_force;              // Device pointer to nodal forces

 protected:

  using Assembly_Interface<T>::M_;
  using Assembly_Interface<T>::K_;
  using Assembly_Interface<T>::F_;

  vector_gpu<T> d_M;   // Device pointer to (diag) matrix M
  vector_gpu<T> d_K;   // Device pointer to matrix K
  vector_gpu<T> d_F;   // Device pointer to vector F

 public:

  // Constructor
  template <template <typename> class MATRIX>
  AssemblyGPU( Problem<T>& p, MATRIX<T>& FEM_Matrix )
      : Assembly_Interface<T>( p, FEM_Matrix ),
      d_coord( p.coord.size() ),
      d_force( p.force.size() ),
      d_M( M_.size() ),
      d_K( K_.size() ),
      d_F( F_.size() ) {}
  // Destructor
  virtual ~AssemblyGPU() {}

  static string name() { return "AssemblyGPU"; }

  // Implement the CPU Interface via the GPU Interface //
  inline void assembleM_cpu() { return assembleM_gpu(); }
  inline void assembleKF_cpu( vector_cpu<T>& coord, vector_cpu<T>& force )
  {
    d_coord = coord; d_force = force;
    return assembleKF_gpu( d_coord, d_force );
  }
  inline void assembleF_cpu( vector_cpu<T>& coord, vector_cpu<T>& force )
  {
    d_coord = coord; d_force = force;
    return assembleF_gpu( d_coord, d_force );
  }
  inline dmatrix<T>&       getM_cpu() { M_ = d_M; return M_; }
  inline matrix_sparse<T>& getK_cpu() { K_ = d_K; return K_; }
  inline vector_cpu<T>&    getF_cpu() { F_ = d_F; return F_; }

  // Define the GPU Interface //
  virtual void assembleM_gpu() = 0;
  virtual void assembleKF_gpu(vector_gpu<T>& coord, vector_gpu<T>& force) = 0;
  virtual void assembleF_gpu(vector_gpu<T>& coord, vector_gpu<T>& force) = 0;

  inline vector_gpu<T>& getM_gpu() { return d_M; }
  inline vector_gpu<T>& getK_gpu() { return d_K; }
  inline vector_gpu<T>& getF_gpu() { return d_F; }
};



#endif
