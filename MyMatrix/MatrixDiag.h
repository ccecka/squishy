#ifndef MATRIXDIAG_H
#define MATRIXDIAG_H

#include "Matrix_Base.h"


template <typename T>
class dmatrix : virtual public matrix_base<T>
{
 protected:

  using matrix_base<T>::n_rows;    // Num Rows

  using matrix_base<T>::val;       // Val array

 public:

  using matrix_base<T>::NOT_STORED;

  // Constructors
  dmatrix() : matrix_base<T>() {}
  dmatrix( int N ) : matrix_base<T>(N,N,N) {}
  dmatrix( int N, T v ) : matrix_base<T>(N,N,N,v) {}
  // Destructor
  virtual ~dmatrix() {}

  static string name() { return "MatDiag"; }

  // Compute y = Ax
  void prod( const vector<T>& x, vector<T>& y ) const
  {
    for( int i = 0; i < n_rows; ++i )
      y[i] = val[i] * x[i];
  }

  // Matrix indexing
  inline int IJtoK( int i, int j ) const
  {
    assert( i >= 0 && i < n_rows && j >= 0 && j < n_rows );
    if( i == j )
      return i;
    else
      return NOT_STORED;
  }

  using matrix_base<T>::operator=;

  // Output
  friend ostream& operator<<( ostream& os, const dmatrix<T>& A )
  {
    ios::fmtflags olda = os.setf(ios::left,ios::adjustfield);
    ios::fmtflags oldf = os.setf(ios::fixed,ios::floatfield);

    int oldp = os.precision(6);

    int ichars = ceil( log10( A.nRows() ) );

    for( int i = 0; i < A.nRows(); ++i ) {
      os << "(";
      os.width( ichars );
      os << i << ",";
      os.width( ichars );
      os << i << "):   " << A(i,i) << endl;
    }

    os.setf(olda,ios::adjustfield);
    os.setf(oldf,ios::floatfield);
    os.precision(oldp);

    return os;
  }

};


#endif
