#ifndef DCSR_MATRIX_H
#define DCSR_MATRIX_H

#include "Matrix_Sparse.h"

#include "MVMRepo.cu"

/* A Condensed Sparse Row formatted M x N matrix
 */


template <typename T>
class matrix_dcsr : public matrix_sparse<T>
{
 protected:

  using matrix_base<T>::n_rows;      // Num Rows
  using matrix_base<T>::n_cols;      // Num Columns

  using matrix_base<T>::val;         // Matrix Entries array

  using matrix_base<T>::NOT_STORED;

  using matrix_sparse<T>::IJ2K;

  vector<int> rowptr;             // Row Pointer array
  vector<int> colidx;             // Column Index array

 public:

  matrix_dcsr() {}
  matrix_dcsr( list< pair<int,int> >& IJ ) { setProfileIJ( IJ ); }
  virtual ~matrix_dcsr() {}

  static string name() { return "DCSR"; }

  inline void setProfileIJ( list< pair<int,int> > IJList )
  {
    IJList.sort();
    IJList.unique();

    vector< pair<int,int> > IJ(IJList.begin(), IJList.end());

    // Determine rows, cols, and nonzeros
    int NZ = IJ.size();
    n_rows = IJ[NZ-1].first + 1;
    n_cols = 0;
    for( int k = 0; k < NZ; ++k ) {
      n_cols = max(n_cols, IJ[k].second+1);
    }

    // This problem must be symmetric!! TODO
    assert( n_cols == n_rows );

    val = vector<T>(NZ,0);
    assert( IJ[0].first == 0 );
    rowptr.resize(n_rows+1);
    colidx.resize(NZ);

    // Store the diagonal NZs first (implictly) in the val array
    int k = 0;
    for( ; k < n_rows; ++k ) {
      IJ2K[ make_pair(k,k) ] = k;
      colidx[k] = k;
    }
    // Store the rest as CSR
    rowptr[0] = k;
    for( int nz = 0; nz < NZ; ++nz ) {
      if( IJ[nz].first != IJ[nz].second ) {
        IJ2K[ IJ[nz] ] = k;
        rowptr[IJ[nz].first+1] = k+1;
        colidx[k] = IJ[nz].second;
        ++k;
      }
    }
    assert( k == NZ );
  }

  /* Accessor Methods */
  inline vector<int>& rowptr_vector() { return rowptr; }
  inline vector<int>& colidx_vector() { return colidx; }


  // Convert the (i,j)th matrix index to the kth index directly into val
  inline int IJtoK( int i, int j ) const
  {
    assert( i >= 0 && i < n_rows && j >= 0 && j < n_rows );

    if( i == j )
      return i;

    // Convert (i,j) to a global int k via binary search
    int first = rowptr[i];
    int last  = rowptr[i+1]-1;
    while( first <= last ) {
      int k = (first + last) / 2;
      int col = colidx[k];
      if( j > col ) {
        first = k + 1;
      } else if( j < col ) {
        last = k - 1;
      } else {  // j == col
        return k;
      }
    }

    // Was not found so return NOT_STORED
    return NOT_STORED;
  }

};


template <typename T>
class DCSR_MVM_CPU : public MVM_CPU<T>
{
 protected:
  using MVM_CPU<T>::h_y;

  int n_rows;

  vector<int> rowptr;
  vector<int> colidx;
 public:

  DCSR_MVM_CPU( matrix_dcsr<T>& A )
      : MVM_CPU<T>(A),
      n_rows(A.nRows()),
      rowptr(A.rowptr_vector()),
      colidx(A.colidx_vector()) {}
  virtual ~DCSR_MVM_CPU() {}
  static string name() { return "DCSR_MVM_CPU"; }

  inline void mvm_cpu( vector_cpu<T>& h_A, vector_cpu<T>& h_x )
  {
    DEBUG_TOTAL(StopWatch timer; timer.start());

    int end = rowptr[0];
    for( int i = 0; i < n_rows; ++i ) {
      T yi = h_A[i] * h_x[i];
      int j = end;               // The new start is the last end
      end = rowptr[i+1];
      for( ; j < end; ++j ) {
        yi += h_A[j] * h_x[colidx[j]];
      }
      h_y[i] = yi;
    }

    INCR_TOTAL(MVM,timer.stop());
  }
};



template <typename T>
class DCSR_MVM_GPU_Scalar : public MVM_GPU<T>
{
 protected:
  using MVM_GPU<T>::d_y;
 public:
  int N;

  const static unsigned int THR_PER_BLOCK = 512;
  const unsigned int NUM_BLOCKS;
  const static bool USE_TEX = false;

  vector_gpu<int> d_row;
  vector_gpu<int> d_col;

  DCSR_MVM_GPU_Scalar( matrix_dcsr<T>& A )
      : MVM_GPU<T>(A),
      N( A.nCols() ),
      NUM_BLOCKS( DIVIDE_INTO(N,THR_PER_BLOCK) ),
      d_row( A.rowptr_vector() ),
      d_col( A.colidx_vector() ) {}
  virtual ~DCSR_MVM_GPU_Scalar() {}
  static string name() { return "DCSR_MVM_GPU_Scalar"; }

  inline void mvm_gpu( vector_gpu<T>& d_A, vector_gpu<T>& d_x )
  {
    DEBUG_TOTAL(StopWatch_GPU timer;  timer.start());

    dcsr_mvm_scalar<THR_PER_BLOCK,USE_TEX>
        <<<NUM_BLOCKS,THR_PER_BLOCK>>>( N, (int*) d_row, (int*) d_col,
                                        d_A, d_x, (T*) d_y );

    INCR_TOTAL(MVM,timer.stop());
  }
};


template <typename T>
class DCSR_MVM_GPU_Vector : public MVM_GPU<T>
{
 protected:
  using MVM_GPU<T>::d_y;
 public:
  int N;

  const static unsigned int THR_PER_VEC   = 32;
  const static unsigned int THR_PER_BLOCK = 256;
  const static unsigned int VEC_PER_BLOCK = THR_PER_BLOCK/THR_PER_VEC;
  const unsigned int NUM_BLOCKS;
  const static bool USE_TEX = false;

  vector_gpu<int> d_row;
  vector_gpu<int> d_col;

  DCSR_MVM_GPU_Vector( matrix_dcsr<T>& A )
      : MVM_GPU<T>(A),
      N( A.nCols() ),
      NUM_BLOCKS( DIVIDE_INTO(N,THR_PER_BLOCK) ),
      d_row( A.rowptr_vector() ),
      d_col( A.colidx_vector() ) {}
  virtual ~DCSR_MVM_GPU_Vector() {}
  static string name() { return "DCSR_MVM_GPU_Vector"; }

  inline void mvm_gpu( vector_gpu<T>& d_A, vector_gpu<T>& d_x )
  {
    DEBUG_TOTAL(StopWatch_GPU timer;  timer.start());

    dcsr_mvm_vector<VEC_PER_BLOCK, THR_PER_VEC, USE_TEX>
        <<<NUM_BLOCKS,THR_PER_BLOCK>>> ( N, (int*) d_row, (int*) d_col,
                                         (T*) d_A, (T*) d_x, (T*) d_y );

    INCR_TOTAL(MVM,timer.stop());
  }
};


#endif
