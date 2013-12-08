#ifndef SOLVER_GPU_H
#define SOLVER_GPU_H

#include "Solver.h"

#include "../MyAssembly/Assembly_Interface.h"

#include "../MyCG/CG_Interface.h"


template <int BLOCK_SIZE, typename T>
__global__ void computeB1( int N, T* b, T* p,
                           T* M, T* K, T* F )
{
  int k = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if( k < N ) {
    T Mk = M[k];
    K[k] += (2/(DT*DT)) * Mk;
    b[k] = (2/DT) * p[k] - F[k];
  }
}

template <int BLOCK_SIZE, typename T>
__global__ void computeB( int N, T* b, T* p,
                          T* coord, T* coordK,
                          T* M, T* K, T* F )
{
  int k = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if( k < N ) {
    T Mk = M[k];
    K[k] += (2/(DT*DT)) * Mk;
    b[k] = (2/DT) * (p[k] - Mk * (coord[k] - coordK[k])/DT) - F[k];
  }
}

template <int BLOCK_SIZE, typename T>
__global__ void updateCoord( int N, T* x, T* dx, T* xP, T* xK )
{
  int k = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if( k < N ) {
    T xk = x[k] + dx[k];
    xP[k] = (xk + xK[k]) / 2;
    x[k] = xk;
  }
}

template <int BLOCK_SIZE, typename T>
__global__ void updateP( int N, T* p,
                         T* M, T* coord, T* coordK,
                         T* F )
{
  int k = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if( k < N )
    p[k] = (1-DAMPING) * (M[k] * (coord[k] - coordK[k]) / DT - (DT/2) * F[k]);
}


template <typename T>
class NR_GPU : public Solver<T>
{
  Assembly_Interface<T>* assembler;
  CG_Interface<T>* cg;

  int N;
  const static int BLOCK_SIZE = 512;
  const unsigned int NUM_BLOCKS;

  vector_gpu<T>& d_M;

  // Coordinate and forcing vectors
  using Solver<T>::d_coord;
  vector_gpu<T> d_coordK;
  vector_gpu<T> d_coordP;
  vector_gpu<T> d_force;

  // Momemtum and scratch vectors
  vector_gpu<T> d_p;
  vector_gpu<T> d_b;

  inline vector_gpu<T>& precomputeM() {
    assembler->assembleM_gpu();
    return assembler->getM_gpu();
  }

 public:

  using Solver<T>::getProblem;
  using Solver<T>::getMesh;

  NR_GPU( Assembly_Interface<T>* assembler_, CG_Interface<T>* cg_ )
      : Solver<T>( assembler_->getProblem() ),
        assembler( assembler_ ),
        cg( cg_ ),
        N( assembler->nEquation() ),
        NUM_BLOCKS( DIVIDE_INTO(N,BLOCK_SIZE) ),
        d_M( precomputeM() ),
        // d_coord = d_vbo on initialization
        d_coordK( getProblem().getCoord().size() ),
        d_coordP( getProblem().getCoord().size() ),
        d_force( getProblem().getForce() ),
        d_p( getProblem().getMomentum() ),
        d_b( N ) {}

  virtual ~NR_GPU() { delete assembler; delete cg; }


  void increment()
  {
    DEBUG_TOTAL(StopWatch_GPU timer;  timer.start());

    d_coordK = d_coord;

    int iter = 0;

    // Assemble
    assembler->assembleKF_gpu( d_coord, d_force );
    vector_gpu<T>& d_K = assembler->getK_gpu();
    vector_gpu<T>& d_F = assembler->getF_gpu();

    // Compute the RHS and make dH from K
    computeB1<BLOCK_SIZE><<<NUM_BLOCKS,BLOCK_SIZE>>>
        (N, (T*) d_b, (T*) d_p, (T*) d_M, (T*) d_K, (T*) d_F);
    CHECKCUDA("computeB1");

    // Compute the norm
    //double normb = cublasSnrm2(N, d_b, 1);
    double normb = norm2( d_b);
    CHECKCUDA("cublassnrm2");
    // Correct normb
    normb *= DT/2;

    //cout << "NR" << ++iter << ": " << normb << endl;

    // Solve the system Kx = b
    cg->CG_gpu( d_K, d_b );

    while( normb > Solver<T>::EPS && iter < Solver<T>::MAX_ITERS ) {

      // Update the coord and get the midpoint
      updateCoord<BLOCK_SIZE><<<NUM_BLOCKS,BLOCK_SIZE>>>
          (N, (T*) d_coord, (T*) cg->getX_gpu(), (T*) d_coordP, (T*) d_coordK);
      CHECKCUDA("midPoint");

      // Assemble
      assembler->assembleKF_gpu( d_coordP, d_force );
      vector_gpu<T>& d_K = assembler->getK_gpu();
      vector_gpu<T>& d_F = assembler->getF_gpu();

      // Compute the RHS and make dH from K
      computeB<BLOCK_SIZE><<<NUM_BLOCKS,BLOCK_SIZE>>>
          (N, (T*) d_b, (T*) d_p, (T*) d_coord, (T*) d_coordK,
           (T*) d_M, (T*) d_K, (T*) d_F);
      CHECKCUDA("computeB");

      // Compute the norm
      normb = norm2( d_b );
      CHECKCUDA("cublassnrm2");
      // Correct normb
      normb *= DT/2;

      cout << "NR" << ++iter << ": " << normb << endl;

      // Solve the system Kx = b
      cg->CG_gpu( d_K, d_b );
    }

    // Update the coord and get the midpoint
    updateCoord<BLOCK_SIZE><<<NUM_BLOCKS,BLOCK_SIZE>>>
        (N, (T*) d_coord, (T*) cg->getX_gpu(), (T*) d_coordP, (T*) d_coordK);
    CHECKCUDA("midPoint");

    // Assemble Force
    assembler->assembleF_gpu( d_coordP, d_force );

    // Update p
    updateP<BLOCK_SIZE><<<NUM_BLOCKS,BLOCK_SIZE>>>
        (N, (T*) d_p, (T*) d_M, (T*) d_coord, (T*) d_coordK,
         (T*) assembler->getF_gpu());
    CHECKCUDA("updateP");

    INCR_TOTAL(NR,timer.stop());
  }

  inline void reset()
  {
    d_coord = getMesh().getCoord();
    d_force.zero();
    d_p.zero();
  }

  inline void setForce( matrix<T>& force )
  {
    d_force = force;
  }

  inline void updateVBO()
  {
    // Nothing to be done
  }
};










#endif
