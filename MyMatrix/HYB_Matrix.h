#ifndef HYB_MATRIX_H
#define HYB_MATRIX_H

#include "Matrix_Sparse.h"

#include "MVMRepo.cu"



template <typename T>
class matrix_hyb : public matrix_sparse<T>
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

  const static int INVALID_INDEX = -1;
  
  matrix_hyb() {} 
  matrix_hyb( list< pair<int,int> >& IJ ) { setProfileIJ( IJ ); }
  virtual ~matrix_hyb() {}
  
  static string name() { return "HYB"; }

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
    float relative_speed = 10.0;
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
    //exit(1);

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
	IJ2K[ IJ[IJ_index] ] = ellidx.IJtoK(k,n);
	ellidx(k,n) = IJ[IJ_index].second;
	++n; 
	++IJ_index;
      }

      // Copy the rest into the COO
      while( k == IJ[IJ_index].first ) {
	IJ2K[ IJ[IJ_index] ] = ellidx.size() + coo_index;
	coorow[coo_index] = k;
	coocol[coo_index] = IJ[IJ_index].second;
	++coo_index;
	++IJ_index;
      }

    }

    val = vector<T>( ellidx.size() + coorow.size(), 0 );
  }

  /* Accessor Methods */
  inline matrix<int,COL_MAJOR>& ell_matrix() { return ellidx; }
  inline vector<int>& coorow_vector() { return coorow; }
  inline vector<int>& coocol_vector() { return coocol; }
 
  // Computes y = Ax
  inline void prod( const vector<T>& x, vector<T>& y) const 
  {
    int coo_index = 0;
    int ellidx_size = ellidx.size();
    int c = INVALID_INDEX;
    for( int row = 0; row < n_rows; ++row ) {
      T yi = 0;
      // Do the ell products
      for( int col = 0; col < ellidx.nCols(); ++col ) {
	if( (c = ellidx(row,col)) == INVALID_INDEX ) break;
	yi += val[ellidx.IJtoK(row,col)] * x[c];
      }
      // Do the coo products
      while( row == coorow[coo_index] ) {
	yi += val[ellidx_size+coo_index] * x[coocol[coo_index]];
	++coo_index;
      }
      y[row] = yi;
    }
  }
 
  // Convert the (i,j)th matrix index to the kth index directly into val
  inline int IJtoK( int i, int j ) const
  {
    if( !(i >= 0 && i < n_rows && j >= 0 && j < n_cols) ) {
      cerr << i << ":" << n_rows << "   " << j << ":" << n_cols << endl;
      assert( i >= 0 && i < n_rows && j >= 0 && j < n_cols );
    }
    
   
    int ellend = ellidx.nCols() - 1;

    if( ellend >= 0 && j == ellidx(i,ellend) ) {
      return ellidx.IJtoK(i,ellend);
    } else if( ellend >= 0 && 
	       (ellidx(i,ellend) == INVALID_INDEX || j < ellidx(i,ellend)) ) {
      // Find it in the ith row of ELL via binary search
      int first = 0;
      int last  = ellidx.nCols() - 2;
      while( first <= last ) {
	int k = (first + last) / 2;
	int col = ellidx(i,k);
	if( col == INVALID_INDEX || j < col ) {
	  last = k - 1;
	} else if( j > col ) {
	  first = k + 1;
	} else {  // j == col
	  return ellidx.IJtoK(i,k);
	}
      }
    } else {
      // Find it in the COO via binary search
      int first = 0;
      int last  = coorow.size() - 1;
      while( first <= last ) {
	int k = (first + last) / 2;
	int row = coorow[k];
	int col = coocol[k];
	if( i < row || (i == row && j < col) ) {
	  last = k - 1;
	} else if( i > row || (i == row && j > col) ) {
	  first = k + 1;
	} else {  // i == row && j == col
	  return ellidx.size() + k;
	}
      }
    }

    // Was not found so return NOT_STORED
    return NOT_STORED;
  }
 
 
  using matrix_base<T>::operator=;
};




template <typename T>
class HYB_MVM_CPU : public MVM_CPU<T>
{
 protected:

  using MVM_CPU<T>::h_y;

 public:

  matrix<int,COL_MAJOR> ellidx;
  vector<int> coorow;
  vector<int> coocol;

  int n_rows;
    
 HYB_MVM_CPU( matrix_hyb<T>& A )
   : MVM_CPU<T>(A),
    ellidx( A.ell_matrix() ),
    coorow( A.coorow_vector() ),
    coocol( A.coocol_vector() ),
    n_rows( A.nRows() ) {}
  virtual ~HYB_MVM_CPU() {}
  static string name() { return "HYB_MVM_CPU"; }
  
  inline void prod_cpu( vector<T>& h_A, vector<T>& h_x )
  {
    DEBUG_TOTAL(StopWatch timer; timer.start());

    int coo_index = 0;
    int ellidx_size = ellidx.size();
    int c = -1;
    for( int row = 0; row < n_rows; ++row ) {
      T yi = 0;
      // Do the ell products
      for( int col = 0; col < ellidx.nCols(); ++col ) {
	if( (c = ellidx(row,col)) == -1 ) break;
	yi += h_A[ellidx.IJtoK(row,col)] * h_x[c];
      }
      // Do the coo products
      for( ; coorow[coo_index] == row; ++coo_index ) {
	yi += h_A[ellidx_size+coo_index] * h_x[coocol[coo_index]];
      }
      h_y[row] = yi;
    }
  }
};




template <typename T>
class HYB_MVM_GPU : public MVM_GPU<T>
{
 protected:
  using MVM_GPU<T>::d_y;
 public:
    
  const static unsigned int BLOCK_SIZE = 256;
  const unsigned int NUM_BLOCKS;
  const static unsigned int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
  const static bool USE_TEX = false;
  
  vector_gpu<int> d_ellidx;
  vector_gpu<int> d_coorow;
  vector_gpu<int> d_coocol;
  
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
  
  
 HYB_MVM_GPU( matrix_hyb<T>& A )
   : MVM_GPU<T>(A),
    NUM_BLOCKS( DIVIDE_INTO(A.nRows(),BLOCK_SIZE) ),
    d_ellidx( (const vector<int>&) A.ell_matrix() ),
    d_coorow( A.coorow_vector() ),
    d_coocol( A.coocol_vector() ),
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
  virtual ~HYB_MVM_GPU() {}
  static string name() { return "HYB_MVM_GPU"; }
  
  inline void prod_gpu( T* d_A, T* d_x )
  {
    DEBUG_TOTAL(StopWatch_GPU timer; timer.start());

    ell_mvm<BLOCK_SIZE, USE_TEX>
      <<<NUM_BLOCKS,BLOCK_SIZE>>>(num_rows, ell_rows, ell_size,
				  (int*) d_ellidx,
				  d_A, d_x, (T*) d_y);
    
    if( num_blocks > 0 ) {

      d_A += ell_size;
      
      coo_mvm_flat<BLOCK_SIZE, USE_TEX>
	<<<num_blocks,BLOCK_SIZE>>>(tail, WARP_SIZE, 
				    (int*) d_coorow, (int*) d_coocol, 
				    d_A, d_x, (T*) d_y,
				    (int*) d_temprows, (T*) d_tempvals);
      
      coo_mvm_reduce<512>
	<<<1,512>>>(num_warps, (int*) d_temprows, (T*) d_tempvals, (T*) d_y);
      
      d_A += tail;
      
      coo_mvm_serial
	<<<1,1>>>(coo_size-tail, 
		  ((int*)d_coorow)+tail, ((int*)d_coocol)+tail, 
		  d_A, d_x, (T*) d_y);
      
    }

    INCR_TOTAL(MVM,timer.stop());
  }
};


#endif
