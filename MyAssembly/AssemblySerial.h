#ifndef ASSEMBLYSERIAL_H
#define ASSEMBLYSERIAL_H

#include "Assembly_Interface.h"

template <typename T>
class AssemblySerial : public AssemblyCPU<T>
{
  using Assembly_Interface<T>::prob;
  using Assembly_Interface<T>::LM;
  
  using Assembly_Interface<T>::M_;
  using Assembly_Interface<T>::K_;
  using Assembly_Interface<T>::F_;

  using Assembly_Interface<T>::arrangeResults;

 public:
  
  // Constructor
  template <template <typename> class MATRIX>
  AssemblySerial( Problem<T>& p, MATRIX<T>& FEM_Matrix ) 
    : AssemblyCPU<T>( p, FEM_Matrix ) {}
  virtual ~AssemblySerial() {}

  static string name() { return "AssemblySerial"; }

  void assembleM_cpu()
  {
    M_.zero();
    dmatrix<T> m_e( prob.nEDOF() );
    
    int nElems = prob.mesh.nElems();
    
    //StopWatch timer; timer.start();

    // Assemble M from elements
    for( int e = 0; e < nElems; ++e ) {
      prob.tetrahedralMass( e, m_e );
      assemble( e, m_e, M_ );
    }

    //double kernel_time = timer.stop();
    //cerr << "AssemblySerialM Kernel: " << kernel_time << endl;
  }

  void assembleKF_cpu( vector_cpu<T>& coord, vector_cpu<T>& f ) 
  {
    K_.zero();
    //F_.zero();
    F_.assign( F_.size(), 0 );

    smatrix<T> k_e( prob.nEDOF(), prob.nEDOF() );
    vector<T>  f_e( prob.nEDOF() );

    int nNodes = prob.mesh.nNodes();
    int nDOF = prob.nNDOF();
    int nElems = prob.mesh.nElems();

    matrix<T> arrangedC( nNodes, nDOF );
    arrangeResults( (vector<T>&) coord, arrangedC );
    matrix<T> arrangedF( nNodes, nDOF  );
    arrangeResults( (vector<T>&) f, arrangedF );

    DEBUG_TOTAL(StopWatch timer; timer.start(););

    // Assemble K and F from elements
    for( int e = 0; e < nElems; ++e ) {
      prob.tetrahedralElasticity( e, arrangedC, arrangedF, k_e, f_e );
      assemble( e, k_e, K_, f_e, F_ );
    }
    
    INCR_TOTAL(AssemblyKF,timer.stop());
  }

  void assembleF_cpu( vector_cpu<T>& coord, vector_cpu<T>& f ) 
  {
    //F_.zero();
    F_.assign( F_.size(), 0 );
    vector<T>  f_e( prob.nEDOF() );
    
    int nNodes = prob.mesh.nNodes();
    int nDOF = prob.nNDOF();
    int nElems = prob.mesh.nElems();

    matrix<T> arrangedC( nNodes, nDOF );
    arrangeResults( (vector<T>&) coord, arrangedC );
    matrix<T> arrangedF( nNodes, nDOF  );
    arrangeResults( (vector<T>&) f, arrangedF );

    DEBUG_TOTAL(StopWatch timer; timer.start(););

    // Assemble F from elements
    for( int e = 0; e < nElems; ++e ) {
      prob.tetrahedralForce( e, arrangedC, arrangedF, f_e );
      assemble( e, f_e, F_ );
    }
    
    INCR_TOTAL(AssemblyF,timer.stop());
  } 

};

#endif
