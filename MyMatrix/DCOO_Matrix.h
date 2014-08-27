#ifndef DCOO_MATRIX_H
#define DCOO_MATRIX_H

#include "Matrix_Sparse.h"

#include "MVMRepo.cu"

/* A sparse (i,j,A) formatted M x N matrix
 * where the first N entries of the val array are
 * reserved for the diagonal entries
 */

template <typename T>
class matrix_dcoo : public matrix_sparse<T>
{
 protected:

  using matrix_base<T>::n_rows;      // Num Rows
  using matrix_base<T>::n_cols;      // Num Columns

  using matrix_base<T>::val;         // Matrix Entries array

  using matrix_base<T>::NOT_STORED;

  using matrix_sparse<T>::IJ2K;

  vector<int> rowidx;             // Row Index array
  vector<int> colidx;             // Col Index array

 public:

  matrix_dcoo() {}
  matrix_dcoo( list< pair<int,int> >& IJ ) { setProfileIJ( IJ ); }
  virtual ~matrix_dcoo() {}
  static string name() { return "DCOO"; }

  inline void setProfileIJ( list< pair<int,int> > IJList )
  {
    IJList.sort();
    IJList.unique();

    vector< pair<int,int> > IJ(IJList.begin(), IJList.end());

    // Determine rows, cols, and nonzeros
    int NZ = IJ.size();
    n_rows = IJ[NZ-1].first + 1;
    n_cols = 0;
    for( int k = 0; k < NZ; ++k ) n_cols = max(n_cols, IJ[k].second+1);

    // This problem must be symmetric!! TODO
    assert( n_cols == n_rows );
    // Store the diagonal NZs first (implictly) in the val array
    rowidx.resize(NZ);
    colidx.resize(NZ);
    val = vector<T>(NZ,0);

    // Fill rowidx and colidx

    // First the diagonal entries
    int k = 0;
    for( ; k < n_rows; ++k ) {
      IJ2K[ make_pair(k,k) ] = k;
      rowidx[k] = k;
      colidx[k] = k;
    }
    // Then the off-diagonal entries
    for( int nz = 0; nz < NZ; ++nz ) {
      if( IJ[nz].first != IJ[nz].second ) {
        IJ2K[ IJ[nz] ] = k;
        rowidx[k] = IJ[nz].first;
        colidx[k] = IJ[nz].second;
        ++k;
      }
    }
    assert( k == NZ );
  }

  /* Accessor Methods */
  inline vector<int>& rowidx_vector() { return rowidx; }
  inline vector<int>& colidx_vector() { return colidx; }

  // Convert the (i,j)th matrix index to the kth index directly into val
  inline int IJtoK( int i, int j ) const
  {
    assert( i >= 0 && i < n_rows && j >= 0 && j < n_cols );

    if( i == j )
      return i;

    // Convert (i,j) to a global int k via binary search
    int first = n_rows;
    int last  = rowidx.size() - 1;
    while( first <= last ) {
      int k = (first + last) / 2;
      int row = rowidx[k];
      int col = colidx[k];
      if( i < row || (i == row && j < col) ) {
        last = k - 1;
      } else if( i > row || (i == row && j > col) ) {
        first = k + 1;
      } else {  // i == row && j == col
        return k;
      }
    }

    // Was not found so return NOT_STORED
    return NOT_STORED;
  }
};


template <typename T>
class DCOO_MVM_CPU : public MVM_CPU<T>
{
 protected:
  using MVM_CPU<T>::h_y;

  int n_rows;

  vector<int> rowidx;
  vector<int> colidx;
 public:

  DCOO_MVM_CPU( matrix_dcoo<T>& A )
      : MVM_CPU<T>(A),
      n_rows(A.nRows()),
      rowidx(A.rowidx_vector()),
      colidx(A.colidx_vector()) {}
  virtual ~DCOO_MVM_CPU() {}
  static string name() { return "DCOO_MVM_CPU"; }

  inline void mvm_cpu( vector_cpu<T>& h_A, vector_cpu<T>& h_x )
  {
    DEBUG_TOTAL(StopWatch timer;  timer.start());

    int nz = n_rows;
    // For each row
    for( int row = 0; row < n_rows; ++row ) {
      T sum = h_A[row] * h_x[row];
      // Add up all NZs that are in this row (assume ordered)
      for( ; rowidx[nz] == row; ++nz )
        sum += h_A[nz] * h_x[colidx[nz]];
      // Store the result
      h_y[row] = sum;
    }

    INCR_TOTAL(MVM,timer.stop());
  }
};

#endif
