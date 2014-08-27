#ifndef SOLVER_H
#define SOLVER_H

#include "../Problem.h"

template <typename T>
class Solver
{
 protected:

  Problem<T>& problem;

  // The vbo copy of the coordinates
  vector_gpu<T> d_coord;

 public:

  const static double EPS = 1e-5;
  const static int MAX_ITERS = 30;

  Solver( Problem<T>& p ) : problem(p) {}
  virtual ~Solver() {}

  inline Problem<T>& getProblem() { return problem; }
  inline Mesh<T>&    getMesh()    { return problem.getMesh(); }

  // Get the current Coords from the vbo
  inline const matrix<T>& getCoord()
  {
    getProblem().getCoord() = d_coord;
    return getProblem().getCoord();
  }
  virtual void setForce( matrix<T>& /*force*/ ) {}

  virtual void reset() {}
  virtual void increment() {}
  inline void initVBO( void* d_vbo )
  {
    d_coord.setPtr( (T*) d_vbo, getProblem().getCoord().size() );
    d_coord = getProblem().getCoord();
  }
  virtual void updateVBO() {}
};


#endif
