#ifndef MATRIX_BASE_H
#define MATRIX_BASE_H

#include "Vector.h"

#include <iostream>

template <typename T>
class matrix_base
{
 protected:

  int n_rows;         // Num Rows
  int n_cols;         // Num Columns
  
  vector_cpu<T> val;      // Val array
  //vector<T> val;

  const static int NOT_STORED = -1;

 public:

  // Constructors
  //matrix_base() : n_rows(0), n_cols(0), val() {};
  //matrix_base(int R, int C) : n_rows(R), n_cols(C), val() {};
  //matrix_base(int R, int C, int N) : n_rows(R), n_cols(C), val(N,0) {}
  explicit matrix_base( int R = 0, int C = 0, int N = 0, T v = T() ) 
    : n_rows(R), n_cols(C), val(N,v) {}
  // Destructor
  virtual ~matrix_base() {}

  static string name() { return "MatBase"; }

  /* Interface */

  // Compute k : val[k] = A(i,j)
  // Returns NOT_STORED if A(i,j) is not stored in val
  // Causes error if (i,j) is out of bounds of A
  virtual int IJtoK( int i, int j ) const = 0;
  

  /* Common Methods */

  //inline void zero() { val.assign( val.size(), 0 ); }
  inline void zero() { val.zero(); }
  inline int nRows() const { return n_rows; }
  inline int nCols() const { return n_cols; }
  inline int size()  const { return val.size(); }
  inline const T operator()( int i, int j ) const 
  {
    int k = IJtoK(i,j);                            // Find the value
    return (k == NOT_STORED ? 0 : val[k]);         // If not found, return 0
  }
  inline T& operator()( int i, int j )
  {
    int k = IJtoK(i,j);                            // Find the value
    assert( k != NOT_STORED );                     // Not found, can't change
    return val[k];
  }
  inline       T& operator[]( int k )       { return val[k]; }
  inline const T& operator[]( int k ) const { return val[k]; }
  // Since the matrix is stored in a single vector allow the cast
  inline operator       vector_cpu<T>& ()       { return val; }
  inline operator const vector_cpu<T>& () const { return val; }
  inline vector_cpu<T>& operator=( const vector_gpu<T>& val_gpu ) 
  { 
    return val = val_gpu; 
  }
  inline vector_cpu<T>& operator=( const vector_cpu<T>& val_cpu ) 
  { 
    return val = val_cpu; 
  }
  
  // Default Output
  friend ostream& operator<<( ostream& os, const matrix_base<T>& A ) 
  {
    ios::fmtflags olda = os.setf(ios::left,ios::adjustfield);
    ios::fmtflags oldf = os.setf(ios::fixed,ios::floatfield);
    
    int oldp = os.precision(6);
    
    int ichars = (int) ceil( log10( A.nRows() ) );
    int jchars = (int) ceil( log10( A.nCols() ) );
   
    for( int i = 0; i < A.nRows(); ++i ) {
      for( int j = 0; j < A.nCols(); ++j ) {
	os << "(";
	os.width( ichars );
	os << i << ",";
	os.width( jchars );
	os << j << "):   " << A(i,j) << endl;
      }
    }

    os.setf(olda,ios::adjustfield);
    os.setf(oldf,ios::floatfield);
    os.precision(oldp);
    
    return os;
  }

};


// Compute y = Ax
template <typename T>
class MVM
{
  int NZ;
  int N;
 protected:
  vector_gpu<T> d_y;
  vector_cpu<T> h_y;
 public:
  MVM( matrix_base<T>& A ) : NZ( A.size() ), N( A.nCols() ), d_y(N), h_y(N) {}
  virtual ~MVM() {}

  static string name() { return "MVM"; }

  inline int mat_size() { return NZ; }
  inline int vec_size() { return N; }

  virtual void mvm_gpu( vector_gpu<T>& d_A, vector_gpu<T>& d_x ) = 0;
  virtual vector_gpu<T>& getY_gpu() = 0;
  virtual void mvm_cpu( vector_cpu<T>& h_A, vector_cpu<T>& h_x ) = 0;
  virtual vector_cpu<T>& getY_cpu() = 0;
};


template <typename T>
class MVM_CPU : public MVM<T>
{
  // Staging space
  vector_cpu<T> h_x;
  vector_cpu<T> h_A;
 protected:
  using MVM<T>::d_y;
  using MVM<T>::h_y;
 public:
  MVM_CPU( matrix_base<T>& A ) : MVM<T>(A), h_x(A.nRows()), h_A(A.size()) {}
  virtual ~MVM_CPU() {}
  static string name() { return "MVM_CPU"; }

  inline void mvm_gpu( vector_gpu<T>& d_A, vector_gpu<T>& d_x )
  {
    h_A = d_A;
    h_x = d_x;
    mvm_cpu( h_A, h_x );
  }
  inline vector_gpu<T>& getY_gpu() { return d_y = h_y; }

  virtual void mvm_cpu( vector_cpu<T>& h_A, vector_cpu<T>& h_x ) = 0;
  inline vector_cpu<T>& getY_cpu() { return h_y; }
};


template <typename T>
class MVM_GPU : public MVM<T>
{
  // Staging space
  vector_gpu<T> d_A;
  vector_gpu<T> d_x;
 protected:
  using MVM<T>::d_y;
  using MVM<T>::h_y;
 public:
 MVM_GPU( matrix_base<T>& A ) : MVM<T>(A), d_A(A.size()), d_x(A.nRows()) {}
  virtual ~MVM_GPU() {}
  static string name() { return "MVM_GPU"; }
  
  virtual void mvm_gpu( vector_gpu<T>& d_A, vector_gpu<T>& d_x ) = 0;
  inline vector_gpu<T>& getY_gpu() { return d_y; }

  inline void mvm_cpu( vector_cpu<T>& h_A, vector_cpu<T>& h_x ) 
  {
    d_A = h_A;
    d_x = h_x;
    mvm_gpu( d_A, d_x );
  }
  inline vector_cpu<T>& getY_cpu() { return h_y = d_y; }
};



#endif
