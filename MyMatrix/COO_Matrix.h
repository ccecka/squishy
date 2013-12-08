#ifndef COO_MATRIX_H
#define COO_MATRIX_H

#include "Matrix_Sparse.h"

#include "MVMRepo.cu"

/* A Condensed Sparse Row formatted M x N matrix
 */

template <typename T>
class matrix_coo : public matrix_sparse<T>
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
 
  matrix_coo() {}
  matrix_coo( list< pair<int,int> >& IJ ) { setProfileIJ( IJ ); }
  virtual ~matrix_coo() {}
  static string name() { return "COO"; }

  inline void setProfileIJ( list< pair<int,int> > IJList )
  {
    IJList.sort();
    IJList.unique();
   
    vector< pair<int,int> > IJ(IJList.begin(), IJList.end());

    int NZ = IJ.size();
    rowidx.resize(NZ);
    colidx.resize(NZ);
    val.resize(NZ);
    val.assign(val.size(), 0);

    n_rows = IJ[NZ-1].first + 1;
    n_cols = 0;
    for( int k = 0; k < NZ; ++k ) {
      IJ2K[ IJ[k] ] = k;
      rowidx[k] = IJ[k].first;
      colidx[k] = IJ[k].second;
      n_cols = max(n_cols, IJ[k].second+1);
    }
  }

  /* Accessor Methods */
  inline vector<int>& rowidx_vector() { return rowidx; }
  inline vector<int>& colidx_vector() { return colidx; }
};


template <typename T>
class COO_MVM_CPU : public MVM_CPU<T>
{
 protected:
  using MVM_CPU<T>::h_y;

  int n_rows;

  vector<int> rowidx;
  vector<int> colidx;
 public:

 COO_MVM_CPU( matrix_coo<T>& A ) 
   : MVM_CPU<T>(A),
    n_rows(A.nRows()),
    rowidx(A.rowidx_vector()),
    colidx(A.colidx_vector()) {}
  virtual ~COO_MVM_CPU() {}
  static string name() { return "COO_MVM_CPU"; }
  
  inline void mvm_cpu( vector_cpu<T>& h_A, vector_cpu<T>& h_x )
  {
    DEBUG_TOTAL(StopWatch timer;  timer.start());

    int nz = 0;
    // For each row
    for( int row = 0; row < n_rows; ++row ) {
      T sum = 0;
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
