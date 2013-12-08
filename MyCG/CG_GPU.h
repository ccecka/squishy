#ifndef CG_GPU_H
#define CG_GPU_H

#include "CG_Interface.h"


template <unsigned int BLOCK_SIZE, typename T>
__global__ void updateR(int N, T* r, T* b, T* Ap)
{
  int t_row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if( t_row < N )
    r[t_row] = b[t_row] - Ap[t_row];
}

template <unsigned int BLOCK_SIZE, typename T>
__global__ void updateX(int N, T* x, T* p, T alpha)
{
  int t_row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if( t_row < N )
    x[t_row] += alpha * p[t_row];
}

template <unsigned int BLOCK_SIZE, typename T>
__global__ void updateXR(int N, T* x, T* p, T* r, T* Ap, T alpha)
{
  int t_row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if( t_row < N ) {
    x[t_row] += alpha *  p[t_row];
    r[t_row] -= alpha * Ap[t_row];
  }
}

template <unsigned int BLOCK_SIZE, typename T>
__global__ void updateP(int N, T* p, T* r, T beta)
{
  int t_row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if( t_row < N )
    p[t_row] = beta * p[t_row] + r[t_row];
}


template <typename T>
class CG_GPU : public CG_GPU_Interface<T>
{
 protected:

  using CG_Interface<T>::mvm;

  using CG_GPU_Interface<T>::d_x;

  vector_gpu<T> d_r;    // Residual vector
  vector_gpu<T> d_p;    // p vector

  const static bool USE_TEX = false;
  const static unsigned int BLOCK_SIZE = 512;
  const unsigned int NUM_BLOCKS;

 public:

  CG_GPU( MVM<T>& mvm )
    : CG_GPU_Interface<T>(mvm),
    d_r(mvm.vec_size()),
    d_p(mvm.vec_size()),
    NUM_BLOCKS( DIVIDE_INTO(mvm.vec_size(),BLOCK_SIZE) ) {}
  virtual ~CG_GPU() {}

  // Define the GPU Interface //
  inline void CG_gpu( vector_gpu<T>& d_A, vector_gpu<T>& d_b )
  {
    DEBUG_TOTAL(StopWatch_GPU timer; timer.start());

    d_x.zero();
    d_r = d_b;
    d_p = d_r;
    CHECKCUDA("GPU CG Alloc");
    
    double rTr_old, alpha, beta;
    //double rTr_new = cublasSdot( d_r.size(), d_r, 1, d_r, 1 );
    double rTr_new = innerProd( d_r, d_r );
    double rTr_eps = rTr_new * CG_Interface<T>::REL_EPS;

    int iter = 0;
    
    // Cuda CG iteration
    while( iter < CG_Interface<T>::MAX_ITER ) {
      mvm.mvm_gpu( d_A, d_p );
      vector_gpu<T>& d_Ap = mvm.getY_gpu();
      CHECKCUDA("GPU CG SpMV");

      //alpha = rTr_new / cublasSdot(d_p.size(), d_p, 1, d_Ap, 1);
      alpha = rTr_new / innerProd( d_p, d_Ap );
      CHECKCUDA("cublasSdot");

      // Residual drift correction
      if( iter % 100 == 0 ) {
	
	// Compute  x += alpha * p  
	updateX<BLOCK_SIZE><<<NUM_BLOCKS,BLOCK_SIZE>>>
	  (d_x.size(), (T*) d_x, (T*) d_p, (T) alpha);
	CHECKCUDA("GPU CG XR");	

	mvm.mvm_gpu( d_A, d_x );
	vector_gpu<T>& d_Ap = mvm.getY_gpu();
	
	// Compute  r = b - Ap
	updateR<BLOCK_SIZE><<<NUM_BLOCKS,BLOCK_SIZE>>>
	  (d_x.size(), (T*) d_r, (T*) d_b, (T*) d_Ap);
	CHECKCUDA("GPU CG XR");

      } else {
	
	// Compute  x += alpha * p  
	//     and  r -= alpha * Ap
	updateXR<BLOCK_SIZE><<<NUM_BLOCKS,BLOCK_SIZE>>>
	  (d_x.size(), (T*) d_x, (T*) d_p, (T*) d_r, (T*) d_Ap, (T) alpha);
	CHECKCUDA("GPU CG XR");
	
      }

      ++iter;
      
      // Compute the new residual innerproduct
      rTr_old = rTr_new;
      //rTr_new = cublasSdot(d_r.size(), d_r, 1, d_r, 1);
      rTr_new = innerProd( d_r, d_r );
      CHECKCUDA("cublasSdot");

      //cout << "CG" << iter << ": " << rTr_new << endl;
      if( rTr_new < rTr_eps || rTr_new < CG_Interface<T>::EPS ) break;
      //if( rTr_new != rTr_new ) {  // rTr_new is a NaN, very wrong
      //cerr << "CG GPU NaN!!" << endl;
      //exit(1);
      //}
      
      // Compute  p = beta*p + r
      beta = rTr_new / rTr_old;
      updateP<BLOCK_SIZE><<<NUM_BLOCKS,BLOCK_SIZE>>>
	(d_p.size(), (T*) d_p, (T*) d_r, (T) beta);
      CHECKCUDA("GPU CG P");    
    }

    INCR_TOTAL(CG,timer.stop());
    //cout << "CG" << iter << ": " << rTr_new << endl;
  }
};



template <unsigned int BLOCK_SIZE, typename T>
__global__ void ixy(int N, T* a, T* b, T* c)
{
  int t_row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if( t_row < N )
    a[t_row] = c[t_row] / b[t_row];
}

template <unsigned int BLOCK_SIZE, typename T>
__global__ void updateRZ(int N, T* r, T* b, T* Ap, T* z, T* A)
{
  int t_row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if( t_row < N ) {
    float ri = b[t_row] - Ap[t_row];
    r[t_row] = ri;
    z[t_row] = (1.0f/A[t_row]) * ri;
  }
}

template <unsigned int BLOCK_SIZE, typename T>
__global__ void updateXRZ(int N, T* x, T* p, T* r, T* Ap, T alpha, T* z, T* A)
{
  int t_row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if( t_row < N ) {
    x[t_row] += alpha *  p[t_row];
    T ri = r[t_row] - alpha * Ap[t_row];
    r[t_row] = ri;
    z[t_row] = ri / A[t_row];
  }
}


template <typename T>
class DCG_GPU : public CG_GPU_Interface<T>
{
 protected:

  using CG_Interface<T>::mvm;

  using CG_GPU_Interface<T>::d_x;

  vector_gpu<T> d_r;    // Residual vector
  vector_gpu<T> d_p;    // p vector
  vector_gpu<T> d_z;

  const static bool USE_TEX = false;
  const static unsigned int BLOCK_SIZE = 512;
  const unsigned int NUM_BLOCKS;

 public:

  DCG_GPU( MVM<T>& mvm )
    : CG_GPU_Interface<T>(mvm),
    d_r(mvm.vec_size()),
    d_p(mvm.vec_size()),
    d_z(mvm.vec_size()),
    NUM_BLOCKS( DIVIDE_INTO(mvm.vec_size(),BLOCK_SIZE) ) {}
  virtual ~DCG_GPU() {}

  // Define the GPU Interface //
  inline void CG_gpu( vector_gpu<T>& d_A, vector_gpu<T>& d_b )
  {
    DEBUG_TOTAL(StopWatch_GPU timer; timer.start(););
    
    d_x.zero();
    d_r = d_b;
    ixy<BLOCK_SIZE><<<NUM_BLOCKS,BLOCK_SIZE>>>
      (d_z.size(), (T*) d_z, (T*) d_A, (T*) d_r);
    d_p = d_z;
    CHECKCUDA("GPU CG Alloc");
    
    double rTz_old, alpha, beta;
    double rTz_new = innerProd( d_r, d_z );   
    double rTz_eps = rTz_new * CG_Interface<T>::REL_EPS;

    int iter = 0;
    
    // Cuda CG iteration
    while( iter < CG_Interface<T>::MAX_ITER ) {
      mvm.mvm_gpu( d_A, d_p );
      vector_gpu<T>& d_Ap = mvm.getY_gpu();
      CHECKCUDA("GPU CG SpMV");

      alpha = rTz_new / innerProd( d_p, d_Ap );

      // Residual drift correction
      if( iter % 100 == 0 ) {
	
	// Compute  x += alpha * p  
	updateX<BLOCK_SIZE><<<NUM_BLOCKS,BLOCK_SIZE>>>
	  (d_x.size(), (T*) d_x, (T*) d_p, (T) alpha);
	CHECKCUDA("GPU CG XR");

	mvm.mvm_gpu( d_A, d_x );
	vector_gpu<T>& d_Ap = mvm.getY_gpu();
	
	// Compute  r = b - Ap
	//     and  z = (1/A) * r
	updateRZ<BLOCK_SIZE><<<NUM_BLOCKS,BLOCK_SIZE>>>
	  (d_x.size(), (T*) d_r, (T*) d_b, (T*) d_Ap, (T*) d_z, (T*) d_A);
	CHECKCUDA("GPU CG XR");

      } else {
	
	// Compute  x += alpha * p  
	//     and  r -= alpha * Ap
	//     and  z = (1/A) * r
	updateXRZ<BLOCK_SIZE><<<NUM_BLOCKS,BLOCK_SIZE>>>
	  (d_x.size(), (T*) d_x, (T*) d_p, (T*) d_r, (T*) d_Ap, 
	   (T) alpha, (T*) d_z, (T*) d_A);
	CHECKCUDA("GPU CG XR");
	
      }

      ++iter;
      
      // Compute the new residual innerproduct
      rTz_old = rTz_new;
      //rTz_new = cublasSdot(d_r.size(), d_r, 1, d_z, 1);
      rTz_new = innerProd( d_r, d_z );
      
      //cout << "CG" << iter << ": " << rTz_new << endl;
      if( rTz_new < rTz_eps || rTz_new < CG_Interface<T>::EPS ) break;
      //if( rTz_new != rTz_new ) {  // rTr_new is a NaN, very wrong
      //cerr << "CG GPU NaN!!" << endl;
      //exit(1);
      //}
      
      // Compute  p = z + beta*p
      beta = rTz_new / rTz_old;
      updateP<BLOCK_SIZE><<<NUM_BLOCKS,BLOCK_SIZE>>>
	(d_p.size(), (T*) d_p, (T*) d_z, (T) beta);
      CHECKCUDA("GPU CG P");    
    }

    INCR_TOTAL(CG,timer.stop());
    //cout << "CG" << iter << ": " << rTz_new << endl;
  }
};









#endif

