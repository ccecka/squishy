#ifndef CG_CPU_H
#define CG_CPU_H

#include "CG_Interface.h"


template <typename T>
class CG_CPU : public CG_CPU_Interface<T>
{
 protected:
  
  using CG_Interface<T>::mvm;

  using CG_CPU_Interface<T>::h_x;

  vector<T> r;
  vector<T> p;
  
 public:
  
 CG_CPU( MVM<T>& mvm )
  : CG_CPU_Interface<T>(mvm),
  r(mvm.vec_size()),
  p(mvm.vec_size()) {}
  virtual ~CG_CPU() {}

  // Define the CPU Interface //
  void CG_cpu( vector_cpu<T>& A, vector_cpu<T>& b )
  {
    DEBUG_TOTAL(StopWatch timer;  timer.start(););

    int N = r.size();

    h_x.zero();
    r = b;
    p = r;

    double rTr_new = innerProd( r, r );
    double rTr_eps = rTr_new * CG_Interface<T>::REL_EPS;

    int iter = 0;
    
    // CG iteration
    while( iter < CG_Interface<T>::MAX_ITER ) {
      
      mvm.mvm_cpu( A, p );
      vector_cpu<T>& Ap = mvm.getY_cpu();

      double alpha = rTr_new / innerProd( p, Ap );
      
      if( iter % 100 == 0 ) {
	
	for( int k = 0; k < N; ++k )
	  h_x[k] += alpha * p[k];
	
	mvm.mvm_cpu( A, h_x );
	vector_cpu<T>& Ap = mvm.getY_cpu();
	
	for( int k = 0; k < N; ++k )
	  r[k] = b[k] - Ap[k];
	
      } else {
	
	for( int k = 0; k < N; ++k ) {
	  h_x[k] += alpha * p[k];
	  r[k] -= alpha * Ap[k];
	}
	
      }
      
      ++iter;
      
      double rTr_old = rTr_new;
      rTr_new = innerProd( r, r );
      
      //cout << "CG" << iter << ": " << rTr_new << endl;
      if( rTr_new < rTr_eps || rTr_new < CG_Interface<T>::EPS ) break;
      //if( rTr_new != rTr_new ) {  // rTr_new is a NaN, very wrong
      //cerr << "CG CPU NaN!!" << endl;
      //exit(1);
      //}

      double beta = rTr_new / rTr_old;
      for( int k = 0; k < N; ++k ) {
	p[k] = r[k] + beta * p[k];
      }
    }

    /*
    // Validate h_x
    mvm.prod_cpu( A, h_x );
    vector<T>& Ap = mvm.getY();
    for( int k = 0; k < N; ++k )
      r[k] = b[k] - Ap[k];
    double rTr = innerProd( r, r );
    cout << "CG Error: " << rTr << endl;
    */

    // Timing
    INCR_TOTAL(CG,timer.stop());
  }
};


// Diagonally preconditioned conjugate gradient
// Assumes the diagonal of the matrix is stored in the first N entries
// of the data array

template <typename T>
class DCG_CPU : public CG_CPU_Interface<T>
{
 protected:
  
  using CG_Interface<T>::mvm;

  using CG_CPU_Interface<T>::h_x;

  vector_cpu<T> r;
  vector_cpu<T> p;
  vector_cpu<T> z;
  
 public:
  
 DCG_CPU( MVM<T>& mvm )
   : CG_CPU_Interface<T>(mvm),
    r(mvm.vec_size()),
    p(mvm.vec_size()),
    z(mvm.vec_size()) {}
  virtual ~DCG_CPU() {}

  // Define the CPU Interface //
  void CG_cpu( vector_cpu<T>& A, vector_cpu<T>& b )
  {
    DEBUG_TOTAL(StopWatch timer;  timer.start(););

    int N = r.size();

    h_x.zero();
    r = b;
    for( int k = 0; k < N; ++k )
      z[k] = (1.0f/A[k]) * r[k];
    p = z;

    double rTz_new = innerProd( r, z );
    double rTz_eps = rTz_new * CG_Interface<T>::REL_EPS;  

    int iter = 0;

    // CG iteration
    while( iter < CG_Interface<T>::MAX_ITER ) {
      
      mvm.mvm_cpu( A, p );
      vector_cpu<T>& Ap = mvm.getY_cpu();

      double alpha = rTz_new / innerProd( p, Ap );
      
      if( iter % 100 == 0 ) {
	
	for( int k = 0; k < N; ++k )
	  h_x[k] += alpha * p[k];
	
	mvm.mvm_cpu( A, h_x );
	vector_cpu<T>& Ap = mvm.getY_cpu();
	
	for( int k = 0; k < N; ++k ) {
	  r[k] = b[k] - Ap[k];
	  z[k] = r[k] / A[k];
	}
	
      } else {
	
	for( int k = 0; k < N; ++k ) {
	  h_x[k] += alpha * p[k];
	  r[k] -= alpha * Ap[k];
	  z[k] = r[k] / A[k];
	}
	
      }
      
      ++iter;
      
      double rTz_old = rTz_new;
      rTz_new = innerProd( r, z );
      
      //cout << "CG" << iter << ": " << rTz_new << endl;
      if( rTz_new < rTz_eps || rTz_new < CG_Interface<T>::EPS ) break;
      //if( rTz_new != rTz_new ) {  // rTr_new is a NaN, very wrong
      //cerr << "CG CPU NaN!!" << endl;
      //exit(1);
      //}
      
      double beta = rTz_new / rTz_old;
      for( int k = 0; k < N; ++k ) {
	p[k] = z[k] + beta * p[k];
      }
    }

    /*
    // Validate h_x
    mvm.prod_cpu( A, h_x );
    vector<T>& Ap = mvm.getY();
    for( int k = 0; k < N; ++k )
      r[k] = b[k] - Ap[k];
    double rTr = innerProd( r, r );
    cout << "CG Error: " << rTr << endl;
    */

    // Timing
    INCR_TOTAL(CG,timer.stop());
  }
};


#endif
