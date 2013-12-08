#ifndef CG_ENV_H
#define CG_ENV_H

#include "../MyMatrix/Vector.h"

template <typename T>
class CG_Interface
{
 protected:

  // The matrix-vector product to use in the CG method
  MVM<T>& mvm;

  // The break conditions
  const static T REL_EPS = 5e-5;   // ||r_k||_2^2 < REL_EPS * ||r_0||_2^2
  const static T EPS = 1e-15;      // ||r_k||_2^2 < EPS
  const int MAX_ITER;              //           k > MAX_ITER

  // Allocated memory for the solution vector
  vector_cpu<T> h_x;
  vector_gpu<T> d_x;
  
 public:
  
  CG_Interface( MVM<T>& mvm_ ) 
    : mvm(mvm_), 
    MAX_ITER((int)sqrt(mvm.vec_size())),
    h_x(mvm.vec_size()), d_x(mvm.vec_size()) {}
  virtual ~CG_Interface() {}

  // Define the CPU Interface //
  virtual void CG_cpu( vector_cpu<T>& h_A, vector_cpu<T>& h_b ) = 0;
  virtual vector_cpu<T>& getX_cpu() = 0;

  // Define the GPU Interface //
  virtual void CG_gpu( vector_gpu<T>& d_A, vector_gpu<T>& d_b ) = 0;
  virtual vector_gpu<T>& getX_gpu() = 0;
};


template <typename T>
class CG_CPU_Interface : public CG_Interface<T>
{
 protected:
  
  using CG_Interface<T>::h_x;
  using CG_Interface<T>::d_x;

 public:
  
 CG_CPU_Interface( MVM<T>& mvm ) : CG_Interface<T>(mvm) {}
  virtual ~CG_CPU_Interface() {}

  // The CPU Interface //
  virtual void CG_cpu( vector_cpu<T>& A, vector_cpu<T>& b ) = 0;
  inline vector_cpu<T>& getX_cpu() { return h_x; }

  // Implement the GPU Interface via the CPU Interface //
  inline void CG_gpu( vector_gpu<T>& d_A, vector_gpu<T>& d_b ) 
  {
    vector_cpu<T> h_A = d_A;
    vector_cpu<T> h_b = d_b;
    CG_cpu( h_A, h_b );
  }
  inline vector_gpu<T>& getX_gpu() { return d_x = h_x; }
};


template <typename T>
class CG_GPU_Interface : public CG_Interface<T>
{
 protected:

  using CG_Interface<T>::h_x;
  using CG_Interface<T>::d_x;

 public:

  CG_GPU_Interface( MVM<T>& mvm ) : CG_Interface<T>(mvm) {}
  virtual ~CG_GPU_Interface() {}

  // Implement the CPU Interface via the GPU Interface //
  inline void CG_cpu( vector_cpu<T>& h_A, vector_cpu<T>& h_b )
  {
    vector_gpu<T> d_A = h_A;
    vector_gpu<T> d_b = h_b;
    CG_gpu( d_A, d_b );
  }
  inline vector_cpu<T>& getX_cpu() { return h_x = d_x; }

  // The GPU Interface //
  virtual void CG_gpu( vector_gpu<T>& d_A, vector_gpu<T>& d_b ) = 0;
  inline vector_gpu<T>& getX_gpu() { return d_x; }
};

#endif

