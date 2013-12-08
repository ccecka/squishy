#ifndef MATRIX_SPARSE_BASE_H
#define MATRIX_SPARSE_BASE_H

#include "Matrix_Base.h"

#include "../Parser.h"

template <typename T>
class matrix_sparse : virtual public matrix_base<T>
{
 protected:

 using matrix_base<T>::n_rows;       // Num Rows
 using matrix_base<T>::n_cols;       // Num Cols
 
 using matrix_base<T>::val;          // Val array
 
 using matrix_base<T>::NOT_STORED;

 public:

 map< pair<int,int>, int > IJ2K;     // (i,j) -> [k] map for all nzs

 // Constructors
 matrix_sparse() : matrix_base<T>() {}
 matrix_sparse(int R, int C) : matrix_base<T>(R,C,0) {}
 matrix_sparse(int R, int C, T v) : matrix_base<T>(R,C,0,v) {}
 // Destructor
 virtual ~matrix_sparse() {}

 static string name() { return "MatSparse"; }

 using matrix_base<T>::operator=;

 inline int nNZ() const
 {
   return IJ2K.size();
 }

 // An easy way to check matrices - reorder val to simple row-major
 virtual vector<T> getCSRA() const 
 {
   vector<T> A(nNZ());
   int nz = 0;
   map< pair<int,int>, int >::const_iterator mi;
   for( mi = IJ2K.begin(); mi != IJ2K.end(); ++mi, ++nz ) {
     A[nz] += val[mi->second];
   }
   return A;
 }

 // Default matrix indexing: use the IJ2K map
 virtual int IJtoK( int i, int j ) const 
 {
   map< pair<int,int>, int >::const_iterator mi = IJ2K.find( make_pair(i,j) );
   if( mi == IJ2K.end() )
     return NOT_STORED;
   return mi->second;
 }

 // Default output: row-major
 friend ostream& operator<<( ostream& os, const matrix_sparse<T>& A ) 
 {
   ios::fmtflags olda = os.setf(ios::right,ios::adjustfield);
   ios::fmtflags oldf = os.setf(ios::scientific,ios::floatfield);
   
   int oldp = os.precision(12);
   
   int ichars = ceil( log10( A.nRows() ) );
   int jchars = ceil( log10( A.nCols() ) );
   
   map< pair<int,int>, int >::const_iterator mi;
   for( mi = A.IJ2K.begin(); mi != A.IJ2K.end(); ++mi ) {
     os << "(";
     os.width( ichars );
     os << mi->first.first << ",";
     os.width( jchars );
     os << mi->first.second << "): ";
     os << A[mi->second] << endl;
   }
   
   os.setf(olda,ios::adjustfield);
   os.setf(oldf,ios::floatfield);
   os.precision(oldp);
   
   return os;
 }

 void writeASCII( const char* filename ) 
 {
   fstream os( filename, ios::out | ios::binary );
   os.setf(ios::right,ios::adjustfield);
   os.setf(ios::scientific,ios::floatfield);
   os.precision(16);
   
   map< pair<int,int>, int >::const_iterator mi;
   for( mi = IJ2K.begin(); mi != IJ2K.end(); ++mi ) {
     os << mi->first.first << " ";
     os << mi->first.second << " ";
     os << val[mi->second] << "\n";
   }
   os.close();
 }

};


// Default Sparse MVM
template <typename T>
class SpMVM_CPU : public MVM_CPU<T>
{
 protected:
  using MVM_CPU<T>::h_y;
  
  map<pair<int,int>, int> IJ2K;

 public:

 SpMVM_CPU( matrix_sparse<T>& A ) 
   : MVM_CPU<T>(A),
    IJ2K(A.IJ2K) {}
  virtual ~SpMVM_CPU() {}
  static string name() { return "SpMVM_CPU"; }
  
  inline void prod_cpu( vector<T>& h_A, vector<T>& h_x )
  {
    h_y.zero();

    map< pair<int,int>, int >::const_iterator mi;
    for( mi = IJ2K.begin(); mi != IJ2K.end(); ++mi ) {
      h_y[mi->first.first] += h_A[mi->second] * h_x[mi->first.second];
    }
  }
};



#endif
