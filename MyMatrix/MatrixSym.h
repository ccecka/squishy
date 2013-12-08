#ifndef MATRIXSYM_H
#define MATRIXSYM_H

#include "Matrix.h"

enum mat_sym { UPPER, LOWER };


template <typename T, mat_order ORDER = ROW_MAJOR, mat_sym SYM = UPPER>
class smatrix : virtual public matrix<T,ORDER>
{
 protected:
  
 using matrix<T>::n_rows;    // Num Rows
 using matrix<T>::n_cols;    // Num Cols
 
 using matrix<T>::val;       // Val array
 
 using matrix<T>::NOT_STORED;

 public:
  
 // Constructors
 smatrix() : matrix_base<T>() {}
 smatrix( int N ) : matrix_base<T>(N,N,(N*(N+1))/2) {}
 smatrix( int N, T v ) : matrix_base<T>(N,N,(N*(N+1))/2,v) {}
 // Destructor
 virtual ~smatrix() {}
  
 static string name() { return "MatSym"; }

 // Matrix indexing
 inline int IJtoK( int i, int j ) const
 {
   assert( i >= 0 && i < n_rows && j >= 0 && j < n_cols );

   // Flip to upper triangular
   if( i > j ) { int t = i; i = j; j = t; }
   
   // ROW UPPER storage is identical to COL LOWER storage
   if( ( ORDER == ROW_MAJOR && SYM == UPPER ) ||
       ( ORDER == COL_MAJOR && SYM == LOWER ) )
     return (i*(2*n_rows-i-1))/2 + j;
   // COL UPPER storage is identical to ROW LOWER storage
   else if( ( ORDER == COL_MAJOR && SYM == UPPER ) ||
	    ( ORDER == ROW_MAJOR && SYM == LOWER ) )
     return (j*(j+1))/2 + i;
   else   // This should never happen
     return -1;
 }
 
};


#endif
