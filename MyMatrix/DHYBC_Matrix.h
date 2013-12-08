#ifndef DHYBC_MATRIX_H
#define DHYBC_MATRIX_H

#include "Matrix_Sparse.h"

#include "MVMRepo.cu"

/* Implements an HYB storiage format, with the assumption that 
 *  every the NZ are in NxN blocks */


template <typename T>
class matrix_dhybc : public matrix_sparse<T>
{ 
 protected:

  const static int N = 3;

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
  
  matrix_dhybc() {}
  matrix_dhybc( list< pair<int,int> >& IJ ) { setProfileIJ( IJ ); }
  virtual ~matrix_dhybc() {}

  static string name() { return "DHYBC"; }
  
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
    
    COUT_VAR(NZ);

    // To verify that the NZs come in NxN blocks, divide all IJ by N
    for( int nz = 0; nz < NZ; ++nz ) {
      IJ[nz] = make_pair( IJ[nz].first / N, IJ[nz].second / N );
    }
    // Make sure there are NxN of each
    sort(IJ.begin(), IJ.end());
    for( int nz = 0; nz < NZ; nz += N*N ) {
      for( int k = nz; k < nz + N*N; ++k ) {
	assert( IJ[nz].first  == IJ[k].first );
	assert( IJ[nz].second == IJ[k].second );
      }
    }

    // Then we can just use these to index the matrix
    unique(IJ.begin(), IJ.end());

    NZ = IJ.size();

    COUT_VAR(NZ);
    for( int nz = 0; nz < NZ; ++nz ) {
      cout << IJ[nz].first << " " << IJ[nz].second << endl;
    }

    // Remove all the diagonal blocks 
    // Reserve space in front for diagNZ
    // Put the rest of the diag block in the COO
    vector< pair<int,int> > offIJ(NZ - n_rows / N);
    list< pair<int,int> > cooIJ;
    int offnz = 0;
    for( int nz = 0; nz < NZ; ++nz ) {
      if( IJ[nz].first != IJ[nz].second ) {

	// This is an offdiag block, save it
	offIJ[offnz] = IJ[nz];
	++offnz;

      } else {

	// Split up this diagonal block
	for( int k1 = 0; k1 < N; ++k1 ) {
	  for( int k2 = 0; k2 < N; ++k2 ) {
	    if( k1 == k2 ) {
	      // This is a diagonal NZ of the matrix
	      int k = N * IJ[nz].first  + k1;
	      IJ2K[ make_pair(k,k) ] = k;
	    } else {
	      // This is an offdiag NZ of the matrix in the diag block
	      int r = N * IJ[nz].first  + k1;
	      int c = N * IJ[nz].second + k2;
	      cooIJ.push_back( make_pair( r, c ) );
	    }
	  }
	}
      }
    }
    IJ = offIJ;
    NZ = IJ.size();

    COUT_VAR(NZ);
    
    // The space reserved for the diagonal NZs
    int DNZ = round_up( n_rows, WARP_SIZE );

    // Determine number of columns per row and maximum
    vector<int> num_cols(n_rows/N, 0);
    for( int k = 0; k < NZ; ++k ) {
      ++num_cols[IJ[k].first];
    }
    int max_cols = max( num_cols );

    cout << num_cols << endl;

    // Compute the distribution of NZs
    vector<int> hist(max_cols + 1, 0);
    for( int k = 0; k < n_rows/N; ++k ) {
      ++hist[ num_cols[k] ];
    }

    //cout << hist << endl;

    // Compute optimal ELL column size
    float relative_speed = 10.0; //3.0;
    int breakeven_threshold = 0; //4096;

    int num_ell_per_row = max_cols;
    for( int k = 0, rows = n_rows/N; k < max_cols; ++k ) {
      rows -= hist[k];    // The number of rows of length > k
      if( relative_speed * rows < n_rows/N || rows < breakeven_threshold ) {
	num_ell_per_row = k;
	break;
      }
    }

    // Compute the number of nonzeros in the ELL and COO portions
    int ell_NZ = 0;
    for( int k = 0; k < n_rows/N; ++k ) {
      ell_NZ += min(num_ell_per_row, num_cols[k]);
    }
    
    // Set sizes
    ellidx = matrix<int,COL_MAJOR>( round_up(n_rows/N,WARP_SIZE), 
				    num_ell_per_row, 
				    INVALID_INDEX );

    COUT_VAR( num_ell_per_row );

    //cout << num_ell_per_row << endl;
    //cout << round_up(n_rows,alignment) << "   " << n_rows << endl;

    // Matrix to help indexing into val
    // An N*N matrix will be stored at each "entry"
    matrix<int,COL_MAJOR> validx_ell( ellidx.nRows(),
				      num_ell_per_row * N*N,
				      INVALID_INDEX );

    // Construct arrays
    int IJ_index = 0;
    for( int k = 0; k < n_rows/N; ++k ) {
      
      // Copy up to num_ell_per_row into the ELL
      int n = 0;
      while( k == IJ[IJ_index].first && n < num_ell_per_row ) {
	//cout << k << " " << n << endl;

	// Define the mapping for the entire block
	for( int k1 = 0; k1 < N; ++k1 ) {
	  for( int k2 = 0; k2 < N; ++k2 ) {
	    int r = N*IJ[IJ_index].first  + k1;
	    int c = N*IJ[IJ_index].second + k2;
	    IJ2K[ make_pair(r,c) ] = DNZ +  
	      validx_ell.IJtoK(k,N*N*n + N*k1 + k2);
	  }
	}
	ellidx(k,n) = IJ[IJ_index].second;
	++n;
	++IJ_index;
      }

      // Copy the rest into the COO
      while( k == IJ[IJ_index].first ) {
	// Wait to define the IJ2K until we sort and define coorow and coocol
	
	// For the entire block
	for( int k1 = 0; k1 < N; ++k1 ) {
	  for( int k2 = 0; k2 < N; ++k2 ) {
	    int r = N*IJ[IJ_index].first  + k1;
	    int c = N*IJ[IJ_index].second + k2;
	    cooIJ.push_back( make_pair( r, c ) );
	  }
	}
	++IJ_index;
      }
    }
    
    cooIJ.sort();
    coorow.resize( cooIJ.size() );
    coocol.resize( cooIJ.size() );
    for( int nz = 0; nz < cooIJ.size(); ++nz ) {
      coorow[nz] = IJ[nz].first;
      coocol[nz] = IJ[nz].second;
      IJ2K[ make_pair( coorow[nz], coocol[nz] ) ] = DNZ + 
	validx_ell.size() + 
	nz;
    }

    val = vector<T>( DNZ + validx_ell.size() + coorow.size() );
  }
 
  /* Accessor Methods */
  inline matrix<int,COL_MAJOR>& ell_matrix() { return ellidx; }
  inline vector<int>& coorow_vector() { return coorow; }
  inline vector<int>& coocol_vector() { return coocol; }
  
  using matrix_base<T>::operator=;

};


#endif
