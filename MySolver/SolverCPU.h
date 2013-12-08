#ifndef SOLVER_CPU_H
#define SOLVER_CPU_H

#include "Solver.h"

#include "../MyAssembly/Assembly_Interface.h"

#include "../MyCG/CG_Interface.h"


template <typename T>
class NR_CPU : public Solver<T>
{
  Assembly_Interface<T>* assembler;
  CG_Interface<T>* cg;

  int N;

  vector_cpu<T>& M;

  // Coordinate and forcing vectors
  vector_cpu<T> coord;
  vector_cpu<T> coordK;
  vector_cpu<T> coordP;
  vector_cpu<T> force;

  // Momemtum and scratch vectors
  vector_cpu<T> p;
  vector_cpu<T> b;

  inline vector_cpu<T>& precomputeM() {
    assembler->assembleM_cpu();
    return assembler->getM_cpu();
  }

 public:

  using Solver<T>::getProblem;
  using Solver<T>::getMesh;

 NR_CPU( Assembly_Interface<T>* assembler_, CG_Interface<T>* cg_ )
   : Solver<T>( assembler_->getProblem() ),
    assembler( assembler_ ),
    cg( cg_ ),
    N( assembler->nEquation() ),
    M( precomputeM() ),
    coord( getProblem().getCoord() ),
    coordK( coord ),
    coordP( coord ),
    force( getProblem().getForce() ),
    p( getProblem().getMomentum() ),
    b( N ) {}

  virtual ~NR_CPU() { delete assembler; delete cg; }

  void increment()
  {
    DEBUG_TOTAL(StopWatch timer;  timer.start());

    coordK = coord;

    int iter = 0;

    double normb = 1;

    while( normb > Solver<T>::EPS && iter < Solver<T>::MAX_ITERS ) {

      // Get the midpoint
      for( int k = 0; k < N; ++k ) {
	coordP[k] = (coord[k] + coordK[k]) / 2;
      }

      // Assemble
      assembler->assembleKF_cpu( coordP, force );
      matrix_sparse<T>& K = assembler->getK_cpu();
      vector_cpu<T>& F = assembler->getF_cpu();

      // Compute the RHS, modify K to construct dH, and norm
      normb = 0;
      for( int k = 0; k < N; ++k ) {
	// Multiply both sides by (2/DT) for efficiency (instead of scaling K)
	K[k] += (2/(DT*DT)) * M[k];
	b[k] = (2/DT) * (p[k] - M[k] * (coord[k] - coordK[k])/DT) - F[k];
	normb += b[k] * b[k];
      }
      // Correct the norm of b
      normb = (DT/2) * sqrt(normb);

      cout << "NR" << ++iter << ": " << normb << endl;

      // Solve the system Kx = b
      cg->CG_cpu( K, b );
      vector_cpu<T>& dx = cg->getX_cpu();

      // Update the coord
      for( int k = 0; k < N; ++k ) {
	coord[k] += dx[k];
      }

    }

    // Get the midpoint
    for( int k = 0; k < N; ++k ) {
      coordP[k] = (coord[k] + coordK[k]) / 2;
    }

    // Assemble Force
    assembler->assembleF_cpu( coordP, force );
    vector_cpu<T>& F = assembler->getF_cpu();

    // Update p
    for( int k = 0; k < N; ++k ) {
      p[k] = (1-DAMPING)*(M[k]*(coord[k]-coordK[k])/DT - (DT/2)*F[k]);
    }

    INCR_TOTAL(NR,timer.stop());
  }

  inline void reset()
  {
    coord = getMesh().getCoord();
    force.zero();
    p.zero();
  }

  inline void setForce( matrix<T>& newForce )
  {
    force = newForce;
  }

  inline void updateVBO()
  {
    Solver<T>::d_coord = coord;
  }
};



#endif
