#ifndef ASSEMBLYSERIALOPT_H
#define ASSEMBLYSERIALOPT_H

#include "Assembly_Interface.h"

template <typename T>
class AssemblySerialOpt : public AssemblyCPU<T>
{
  using Assembly_Interface<T>::prob;
  using Assembly_Interface<T>::LM;

  using Assembly_Interface<T>::M_;
  using Assembly_Interface<T>::K_;
  using Assembly_Interface<T>::F_;

  // Empty (diagonal) element lumped mass matrix
  dmatrix<T> m_e_;
  // Empty element stiffness matrix
  smatrix<T> k_e_;
  // Empty element forcing vector
  vector<T> f_e_;
  
  using Assembly_Interface<T>::arrangeResults;
  
  matrix<int> LMk;

 public:
  
  // Constructor
  template <template <typename> class MATRIX>
  AssemblySerialOpt( Problem<T>& p, MATRIX<T>& FEM_Matrix ) 
    : AssemblyCPU<T>( p, FEM_Matrix ),
    LMk( prob.mesh.nElems(), prob.nEDOF()*prob.nEDOF()+2*prob.nEDOF(), -1 ),
    m_e_( prob.nEDOF() ),
    k_e_( prob.nEDOF(), prob.nEDOF() ),
    f_e_( prob.nEDOF() )
      {
	int nElems = prob.mesh.nElems();
	int eDoF = prob.nEDOF();

	// For each element
	for( int e = 0; e < nElems; ++e ) {
	  
	  // Pre compute the LMk map for F
	  for( int k1 = 0; k1 < eDoF; ++k1 ) {
	    int r = LM(e,k1);
	    if( r == -1 ) continue;
	    
	    LMk(e,k1) = r;
	  }
	  
	  int LMindex = eDoF;

	  // Precompute the LMk map for K
	  for( int k1 = 0; k1 < eDoF; ++k1 ) {
	    int r = LM(e,k1);
	    if( r == -1 ) continue;
	    
	    for( int k2 = k1; k2 < eDoF; ++k2 ) {
	      int c = LM(e,k2);
	      if( c == -1 ) continue;
	      
	      LMk(e,LMindex++) = K_.IJtoK( r, c );
	      LMk(e,LMindex++) = K_.IJtoK( c, r );
	    }
	  }
	}
      }
  virtual ~AssemblySerialOpt() {}
  
  static string name() { return "AssemblySerialOpt"; }


  void assembleM_cpu()
  {
    M_.zero();
    
    int nElems = prob.mesh.nElems();

    int Nke = k_e_.size();
    int Nfe = f_e_.size();

    //StopWatch timer; timer.start();

    // Assemble K and F from elements
    for( int e = 0; e < nElems; ++e ) {
      //cout << e << endl;
      prob.tetrahedralMass( e, m_e_ );
      
      for( int k = 0; k < Nfe; ++k ) {
	M_[LMk(e,k)] += m_e_[k];
      }
    }

    //double kernel_time = timer.stop();
    //cerr << "AssemblySerialOptM Kernel: " << kernel_time << endl;
  }

  void assembleKF_cpu( vector_cpu<T>& coord, vector_cpu<T>& f ) 
  {
    K_.zero();
    F_.zero();

    int nNodes = prob.mesh.nNodes();
    int nDOF = prob.nNDOF();
    int nElems = prob.mesh.nElems();
    int eDoF = prob.nEDOF();

    int Nke = k_e_.size();
    int Nfe = f_e_.size();

    matrix<T> arrangedC( nNodes, nDOF );
    arrangeResults( (vector<T>&) coord, arrangedC );
    matrix<T> arrangedF( nNodes, nDOF  );
    arrangeResults( (vector<T>&) f, arrangedF );

    int LMindex = 0;

    DEBUG_TOTAL(StopWatch timer; timer.start(););
      
    // Assemble K and F from elements
    for( int e = 0; e < nElems; ++e ) {
      prob.tetrahedralElasticity( e, arrangedC, arrangedF, k_e_, f_e_ );
      
      // Assemble using our LMk for direct indexing
      for( int k = 0; k < Nfe; ++k ) {
	F_[LMk[LMindex++]] += f_e_[k];
      }

      // Do each twice due to symmetry (sorry diagonals)
      for( int k = 0; k < Nke; ++k ) {
	T kek = k_e_[k];

	int indexIJ = LMk[LMindex++];
	K_[indexIJ] += kek;

	int indexJI = LMk[LMindex++];
	if( indexJI != indexIJ )
	  K_[indexJI] += kek;
      }
    }

    INCR_TOTAL(AssemblyKF,timer.stop());
  }

  void assembleF_cpu( vector_cpu<T>& coord, vector_cpu<T>& f ) 
  {
    F_.zero();
        
    int nNodes = prob.mesh.nNodes();
    int nDOF = prob.nNDOF();
    int nElems = prob.mesh.nElems();

    int Nke = k_e_.size();
    int Nfe = f_e_.size();

    matrix<T> arrangedC( nNodes, nDOF );
    arrangeResults( (vector<T>&) coord, arrangedC );
    matrix<T> arrangedF( nNodes, nDOF  );
    arrangeResults( (vector<T>&) f, arrangedF );

    DEBUG_TOTAL(StopWatch timer; timer.start(););

    // Assemble K and F from elements
    for( int e = 0; e < nElems; ++e ) {
      prob.tetrahedralForce( e, arrangedC, arrangedF, f_e_ );
      
      for( int k = 0; k < Nfe; ++k ) {
	F_[LMk(e,k)] += f_e_[k];
      }
    }

    INCR_TOTAL(AssemblyF,timer.stop());
  }

};


#endif
