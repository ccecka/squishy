#ifndef DHYB_MATRIX_H
#define DHYB_MATRIX_H

#include "Matrix_Sparse.h"

#include "MVMRepo.cu"


template <typename T>
class matrix_dhyb : public matrix_sparse<T>
{
 protected:

  using matrix_base<T>::n_rows;      // Num Rows
  using matrix_base<T>::n_cols;      // Num Columns

  using matrix_base<T>::val;         // Matrix Entries array

  using matrix_base<T>::NOT_STORED;

  using matrix_sparse<T>::IJ2K;

  matrix<int,COL_MAJOR> ellidx;

  vector<int> coorow;
  vector<int> coocol;

 public:

  int DNZ;

  const static int INVALID_INDEX = -1;

  matrix_dhyb() {}
  matrix_dhyb( list< pair<int,int> >& IJ ) { setProfileIJ( IJ ); }
  virtual ~matrix_dhyb() {}

  static string name() { return "DHYB"; }

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
    DNZ = round_up( n_rows, WARP_SIZE );

    // Remove all the diagonal NZs and reserve space in front for them
    vector< pair<int,int> > offIJ(NZ - n_rows);
    int offnz = 0;
    for( int nz = 0; nz < NZ; ++nz ) {
      if( IJ[nz].first != IJ[nz].second ) {
	offIJ[offnz] = IJ[nz];
	++offnz;
      } else {
	IJ2K[ IJ[nz] ] = IJ[nz].first;
      }
    }
    IJ = offIJ;
    NZ = IJ.size();

    // Determine number of columns per row and maximum
    vector<int> num_cols(n_rows,0);
    for( int k = 0; k < NZ; ++k ) {
      ++num_cols[IJ[k].first];
    }
    int max_cols = max( num_cols );

    // Compute the distribution of NZs
    vector<int> hist(max_cols + 1, 0);
    for( int k = 0; k < n_rows; ++k ) {
      ++hist[ num_cols[k] ];
    }

    //cout << hist << endl;

    // Compute optimal ELL column size
    float relative_speed = 10.0; //3.0;
    int breakeven_threshold = 0; //4096;

    int num_ell_per_row = max_cols;
    for( int k = 0, rows = n_rows; k < max_cols; ++k ) {
      rows -= hist[k];    // The number of rows of length > k
      if( relative_speed * rows < n_rows || rows < breakeven_threshold ) {
	num_ell_per_row = k;
	break;
      }
    }

    // Compute the number of nonzeros in the ELL and COO portions
    int ell_NZ = 0;
    for( int k = 0; k < n_rows; ++k ) {
      ell_NZ += min(num_ell_per_row, num_cols[k]);
    }
    int coo_NZ = NZ - ell_NZ;

    // Set sizes
    ellidx = matrix<int,COL_MAJOR>( round_up(n_rows,WARP_SIZE),
				    num_ell_per_row,
				    INVALID_INDEX );
    coorow.resize( coo_NZ );
    coocol.resize( coo_NZ );

    COUT_VAR( num_ell_per_row );

    //cout << num_ell_per_row << endl;
    //cout << round_up(n_rows,alignment) << "   " << n_rows << endl;

    // Construct arrays
    int IJ_index = 0;
    int coo_index = 0;
    for( int k = 0; k < n_rows; ++k ) {

      // Copy up to num_ell_per_row into the ELL
      int n = 0;
      while( k == IJ[IJ_index].first && n < num_ell_per_row ) {
	//cout << k << " " << n << endl;
	IJ2K[ IJ[IJ_index] ] = DNZ + ellidx.IJtoK(k,n);
	ellidx(k,n) = IJ[IJ_index].second;
	++n;
	++IJ_index;
      }

      // Copy the rest into the COO
      while( k == IJ[IJ_index].first ) {
	IJ2K[ IJ[IJ_index] ] = DNZ + ellidx.size() + coo_index;
	coorow[coo_index] = k;
	coocol[coo_index] = IJ[IJ_index].second;
	++coo_index;
	++IJ_index;
      }

    }

    val = vector<T>( DNZ + ellidx.size() + coorow.size(), 0 );
  }

  /* Accessor Methods */
  inline matrix<int,COL_MAJOR>& ell_matrix() { return ellidx; }
  inline vector<int>& coorow_vector() { return coorow; }
  inline vector<int>& coocol_vector() { return coocol; }

};




template <typename T>
class DHYB_MVM_CPU : public MVM_CPU<T>
{
 protected:

  using MVM_CPU<T>::h_y;

 public:

  matrix<int,COL_MAJOR> ellidx;
  vector<int> coorow;
  vector<int> coocol;

  int n_rows;
  int DNZ;

  DHYB_MVM_CPU( matrix_dhyb<T>& A )
      : MVM_CPU<T>(A),
      ellidx( A.ell_matrix() ),
      coorow( A.coorow_vector() ),
      coocol( A.coocol_vector() ),
      n_rows( A.nRows() ),
      DNZ(A.DNZ) {}
  virtual ~DHYB_MVM_CPU() {}
  static string name() { return "DHYB_MVM_CPU"; }

  inline void mvm_cpu( vector_cpu<T>& h_A, vector_cpu<T>& h_x )
  {
    DEBUG_TOTAL(StopWatch_GPU timer; timer.start());

    int coo_index = 0;
    int ellidx_size = ellidx.size();
    int c = -1;
    for( int row = 0; row < n_rows; ++row ) {
      T yi = h_A[row] * h_x[row];
      // Do the ell products
      for( int col = 0; col < ellidx.nCols(); ++col ) {
	if( (c = ellidx(row,col)) == -1 ) break;
	yi += h_A[DNZ+ellidx.IJtoK(row,col)] * h_x[c];
      }
      // Do the coo products
      for( ; coorow[coo_index] == row; ++coo_index ) {
	yi += h_A[DNZ+ellidx_size+coo_index] * h_x[coocol[coo_index]];
      }
      h_y[row] = yi;
    }

    INCR_TOTAL(MVM,timer.stop());
  }
};





template <typename T>
class DHYB_MVM_GPU : public MVM_GPU<T>
{
 protected:
  using MVM_GPU<T>::d_y;
 public:

  const static unsigned int BLOCK_SIZE = 512;
  const unsigned int NUM_BLOCKS;
  const static unsigned int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
  const static bool USE_TEX = false;

  vector_gpu<int> d_ellidx;
  vector_gpu<int> d_coorow;
  vector_gpu<int> d_coocol;

  int DNZ;

  int num_rows;
  int ell_rows;
  int ell_cols;

  int ell_size;
  int coo_size;

  int num_warps;
  int num_blocks;
  int tail;

  vector_gpu<int> d_temprows;
  vector_gpu<T> d_tempvals;


  DHYB_MVM_GPU( matrix_dhyb<T>& A )
      : MVM_GPU<T>(A),
      NUM_BLOCKS( DIVIDE_INTO(A.nRows(),BLOCK_SIZE) ),
      d_ellidx( A.ell_matrix() ),
      d_coorow( A.coorow_vector() ),
      d_coocol( A.coocol_vector() ),
      DNZ(A.DNZ),
      num_rows( A.nRows() ),
      ell_rows( A.ell_matrix().nRows() ),
      ell_cols( A.ell_matrix().nCols() ),
      ell_size( A.ell_matrix().size() ),
      coo_size( A.coorow_vector().size() ),
      num_warps( A.coorow_vector().size() / WARP_SIZE ),
      num_blocks( DIVIDE_INTO(num_warps, WARPS_PER_BLOCK) ),
    tail( num_warps * WARP_SIZE ),
    d_temprows( num_warps ),
    d_tempvals( num_warps ) {}

  virtual ~DHYB_MVM_GPU() {}
  static string name() { return "DHYB_MVM_GPU"; }

  inline void mvm_gpu( vector_gpu<T>& d_A, vector_gpu<T>& d_x )
  {
    DEBUG_TOTAL(StopWatch_GPU timer; timer.start());

    T* d_A_ptr = (T*) d_A;

    dell_mvm<BLOCK_SIZE, USE_TEX>
        <<<NUM_BLOCKS,BLOCK_SIZE>>>(DNZ, num_rows, ell_rows, ell_size,
                                    (int*) d_ellidx,
                                    d_A_ptr, (T*) d_x, (T*) d_y);

    if( num_blocks > 0 ) {

      d_A_ptr += DNZ + ell_size;

      coo_mvm_flat<BLOCK_SIZE, USE_TEX>
          <<<num_blocks,BLOCK_SIZE>>>(tail, WARP_SIZE,
                                      (int*) d_coorow, (int*) d_coocol,
                                      d_A_ptr, (T*) d_x, (T*) d_y,
                                      (int*) d_temprows, (T*) d_tempvals);

      coo_mvm_reduce<512>
          <<<1,512>>>(num_warps, (int*) d_temprows, (T*) d_tempvals, (T*) d_y);

      d_A_ptr += tail;

      coo_mvm_serial
          <<<1,1>>>(coo_size-tail,
                    ((int*)d_coorow)+tail, ((int*)d_coocol)+tail,
                    d_A_ptr, (T*) d_x, (T*) d_y);

    }

    INCR_TOTAL(MVM,timer.stop());
  }
};



#endif
