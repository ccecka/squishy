#ifndef MATRIX_H
#define MATRIX_H

#include "Matrix_Base.h"

enum mat_order { ROW_MAJOR, COL_MAJOR };


template <typename T, mat_order ORDER = ROW_MAJOR>
class matrix : virtual public matrix_base<T>
{
 protected:
 
 using matrix_base<T>::n_rows;    // Num Rows
 using matrix_base<T>::n_cols;    // Num Cols
 
 using matrix_base<T>::val;       // Val array
 
 using matrix_base<T>::NOT_STORED;

 public:
  
 // Constructors
 matrix() : matrix_base<T>() {}
 matrix(int R, int C) : matrix_base<T>(R,C,R*C) {}
 matrix(int R, int C, T v) : matrix_base<T>(R,C,R*C,v) {}
 // Destructor
 virtual ~matrix() {}

 static string name() { return "Matrix"; }

 using matrix_base<T>::operator=;

 // Compute y = Ax
 virtual void prod( const vector<T>& x, vector<T>& y ) const
 {
   for( int i = 0; i < n_rows; ++i ) {
     T yi = 0;
     for( int j = 0; j < n_cols; ++j )
       yi += (*this)(i,j) * x[j];
     y[i] = yi;
   }
 }
 
 // Matrix indexing
 virtual int IJtoK( int i, int j ) const
 {
   assert( i >= 0 && i < n_rows && j >= 0 && j < n_cols );
   
   switch( ORDER ) {
   case ROW_MAJOR: return n_cols*i + j;
   case COL_MAJOR: return i + n_rows*j;
   // This should never happen
   default:        return -1;
   }
 }
 
};

#endif
