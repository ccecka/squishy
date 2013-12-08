#ifndef VECTOR_H
#define VECTOR_H

#include "../General.cu"

#include <vector>
#include <algorithm>

template <class T>
ostream& operator<<(ostream& os, const vector<T>& a)
{
  ios::fmtflags olda = os.setf(ios::right,ios::adjustfield);
  ios::fmtflags oldf = os.setf(ios::scientific,ios::floatfield);

  int oldp = os.precision(8);

  for( int k = 0; k < a.size(); ++k )
    os << k << ":  " << a[k] << endl;

  os.setf(olda,ios::adjustfield);
  os.setf(oldf,ios::floatfield);
  os.precision(oldp);
  os << "";

  return os;
}



template <typename T> class vector_gpu;

// Simple storage on the CPU
template <typename T>
class vector_cpu
{
 protected:

  friend class vector_gpu<T>;

  vector<T> h_v;

 public:

  explicit vector_cpu( int N = 0, T v = T() ) : h_v(N,v)
  {
    //cerr << "VCPU Nv Constructor " << N << "  " << v << endl;
  }
  vector_cpu( const vector<T>& a ) : h_v(a)
  {
    //cerr << "VCPU(V) Copy Constructor" << endl;
  }
  vector_cpu( const vector_cpu<T>& a ) : h_v(a.h_v)
  {
    //cerr << "VCPU(VCPU) Copy Constructor" << endl;
  }
  vector_cpu( const vector_gpu<T>& a ) : h_v(a.size())
  {
    //cerr << "VCPU(VGPU) Copy Constructor" << endl;
    cudaMemcpyD2H( h_v, (const T*) a );
  }
  ~vector_cpu()
  {
    //cerr << "VCPU Destructor" << endl;
  }

  inline void zero() { h_v.assign( size(), 0 ); }
  inline int size() const { return h_v.size(); }

  inline       T& operator[]( int k )       { return h_v[k]; }
  inline const T& operator[]( int k ) const { return h_v[k]; }

  // Cast to an stl vector
  //inline operator       vector<T>&()       { return h_v; }
  //inline operator const vector<T>&() const { return h_v; }

  // Copy by Value
  inline vector_cpu<T>& operator=( const vector<T>& rhs )
  {
    //cerr << "VCPU = V" << endl;
    h_v = rhs;
    return *this;
  }
  // Copy by Value
  inline vector_cpu<T>& operator=( const vector_cpu<T>& rhs )
  {
    //cerr << "VCPU = VCPU" << endl;
    h_v = rhs.h_v;
    return *this;
  }
  // Copy by Value
  inline vector_cpu<T>& operator=( const vector_gpu<T>& rhs )
  {
    //cerr << "VCPU = VGPU " << rhs.size() << endl;
    h_v.resize( rhs.size() );
    cudaMemcpyD2H( h_v, (const T*) rhs );
    return *this;
  }

  friend ostream& operator<<(ostream& os, const vector_cpu<T>& a)
  {
    return os << a.h_v;
  }
};


// A deep copy gpu vector that cleans up after itself when descoped //
template <typename T>
class vector_gpu
{
 protected:

  friend class vector_cpu<T>;

  T* d_v;
  int N;
  bool destruct;

 public:

  explicit vector_gpu( int N_ = 0 ) : d_v(cudaNew<T>(N_)), N(N_), destruct(true)
  {
    //cerr << "VGPU N Constructor" << endl;
  }
  vector_gpu( const vector<T>& v )
      : d_v(cudaNew(v)), N(v.size()), destruct(true)
  {
    //cerr << "VGPU(V) Copy Constructor" << endl;
  }
  vector_gpu( const vector_cpu<T>& v )
      : d_v(cudaNew(v.h_v)), N(v.size()), destruct(true)
  {
    //cerr << "VGPU(VCPU) Copy Constructor" << endl;
  }
  vector_gpu( const vector_gpu<T>& v )
      : d_v(cudaNew<T>(v.size())), N(v.size()), destruct(true)
  {
    //cerr << "VGPU(VGPU) Copy Constructor" << endl;
    cudaMemcpyD2D( d_v, (const T*) v, N );
  }
  ~vector_gpu()
  {
    //cerr << "VGPU Destructor" << endl;
    if( destruct ) cudaDelete( d_v );
  }

  inline void zero() { cudaMemset( d_v, 0, N*sizeof(T) ); }
  inline int size() const { return N; }

  // Cast to device pointer
  inline operator       T*()       { return d_v; }
  inline operator const T*() const { return d_v; }

  // HACK!! A way to cheat and do a hidden shallow copy
  inline vector_gpu<T>& setPtr( T* d_v_, int N_ )
  {
    if( destruct ) cudaDelete( d_v );
    d_v = d_v_;  N = N_;  destruct = false;
    return *this;
  }

  // Copy by Value
  inline vector_gpu<T>& operator=( const vector<T>& rhs )
  {
    //cerr << "VGPU = V" << endl;
    if( size() != rhs.size() ) {
      if( destruct ) cudaDelete( d_v );
      N = rhs.size();
      d_v = cudaNew<T>(N);
    }
    cudaMemcpyH2D( d_v, rhs );
    return *this;
  }
  // Copy by Value
  inline vector_gpu<T>& operator=( const vector_cpu<T>& rhs )
  {
    //cerr << "VGPU = VCPU" << endl;
    if( size() != rhs.size() ) {
      if( destruct ) cudaDelete( d_v );
      N = rhs.size();
      d_v = cudaNew<T>(N);
    }
    cudaMemcpyH2D( d_v, rhs.h_v );
    return *this;
  }
  // Copy by Value
  inline vector_gpu<T>& operator=( const vector_gpu<T>& rhs )
  {
    //cerr << "VGPU = VGPU" << endl;
    if( size() != rhs.size() ) {
      if( destruct ) cudaDelete( d_v );
      N = rhs.size();
      d_v = cudaNew<T>(N);
    }
    cudaMemcpyD2D( d_v, rhs.d_v, size() );
    return *this;
  }

  friend ostream& operator<<(ostream& os, const vector_gpu<T>& a)
  {
    vector_cpu<T> temp = a;
    return os << temp;
  }
};


template <typename T>
inline T sum( const vector<T>& a ) {
  int N = a.size();
  T sumA = a[0];
  for( int k = 1; k < N; ++k ) {
    sumA += a[k];
  }
  return sumA;
}

template <typename T>
inline T max( const vector<T>& a ) {
  int N = a.size();
  T maxA = a[0];
  for( int k = 1; k < N; ++k ) {
    if( a[k] > maxA )
      maxA = a[k];
  }
  return maxA;
}

template <typename T>
inline T innerProd( const vector<T>& a, const vector<T>& b )
{
  int N = a.size();
  double sum = 0;
  for( int k = 0; k < N; ++k )
    sum += a[k] * b[k];
  return (T) sum;
}

template <typename T>
inline T innerProd( const vector_cpu<T>& a, const vector_cpu<T>& b )
{
  return innerProd( (const vector<T>&) a, (const vector<T>&) b );
}

inline float innerProd( const vector_gpu<float>& a,
                        const vector_gpu<float>& b )
{
  return cublasSdot( a.size(), (const float*) a, 1, (const float*) b, 1 );
}

inline double innerProd( const vector_gpu<double>& a,
                         const vector_gpu<double>& b )
{
  return cublasDdot( a.size(), (const double*) a, 1, (const double*) b, 1 );
}



template <typename T>
inline T norm2( const vector<T>& a )
{
  int N = a.size();
  double sum = 0;
  for( int k = 0; k < N; ++k )
    sum += a[k] * a[k];
  return (T) sqrt(sum);
}

template <typename T>
inline T norm2( const vector_cpu<T>& a )
{
  return norm2( (const vector<T>&) a );
}

inline float norm2( const vector_gpu<float>& a )
{
  return cublasSnrm2( a.size(), (const float*) a, 1 );
}

inline double norm2( const vector_gpu<double>& a )
{
  return cublasDnrm2( a.size(), (const double*) a, 1 );
}

#endif
