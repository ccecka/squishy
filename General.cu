#ifndef CUDA_GENERAL_CU
#define CUDA_GENERAL_CU

/** A file of common CUDA includes and simple functions **/
/** to be directly #included by other cuda files.       **/

#include <cublas.h>

#include "General.h"

// Define to check for errors after each cuda call
//#define DEBUG_CUDA

#ifdef DEBUG_CUDA
#define CHECKCUDA(s) {cudaError_t err = cudaThreadSynchronize(); \
                      if( err != cudaSuccess ) { \
                        cerr << "CUDA " << s \
                             << ": " << cudaGetErrorString(err) << endl; \
                        exit(1); \
                      }}
#else
#define CHECKCUDA(s)
#endif

#define SAFECUDA(s) s;CHECKCUDA(#s)


// A quick class to time gpu kernels using device events
struct StopWatch_GPU
{
  cudaEvent_t startTime, stopTime;
  StopWatch_GPU()  { cudaEventCreate(&startTime); cudaEventCreate(&stopTime); }
  ~StopWatch_GPU() { cudaEventDestroy(startTime); cudaEventDestroy(stopTime); }
  inline void start() { cudaEventRecord(startTime,0); }
  inline double stop() { return elapsed(); }
  inline double elapsed() 
  {
    cudaEventRecord(stopTime,0);
    cudaEventSynchronize(stopTime);
    float result;
    cudaEventElapsedTime(&result, startTime, stopTime);
    return result/1000.0;    // 1000 mSec per Sec
  }
};





inline int cudaMaxSMEM()
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  return deviceProp.sharedMemPerBlock;
}


template <class T>
inline void cudaMemcpyD2D( T* d_a, const T* d_b, int N )
{
  DEBUG_TOTAL(StopWatch_GPU timer;  timer.start());
  SAFECUDA(cudaMemcpy( d_a, d_b, N*sizeof(T), cudaMemcpyDeviceToDevice ))
  //CHECKCUDA("MemcpyD2D Error");
  INCR_TOTAL(Transfer,timer.stop());
  //cout << "D2D: " << N << endl;
}

template <class T>
inline void cudaMemcpyH2D( T* d_a, const T* h_a, int N )
{
  DEBUG_TOTAL(StopWatch_GPU timer;  timer.start());
  cudaMemcpy( d_a, h_a, N*sizeof(T), cudaMemcpyHostToDevice );
  CHECKCUDA("MemcpyH2D Error");
  INCR_TOTAL(Transfer,timer.stop());
  //cout << "H2D: " << N << endl;
}

template <class T>
inline void cudaMemcpyD2H( T* h_a, const T* d_a, int N )
{
  DEBUG_TOTAL(StopWatch_GPU timer;  timer.start());
  cudaMemcpy( h_a, d_a, N*sizeof(T), cudaMemcpyDeviceToHost );
  CHECKCUDA("MemcpyD2H Error");
  INCR_TOTAL(Transfer,timer.stop());
  //cout << "D2H: " << N << endl;
}



template <class T>
inline void cudaMemcpyH2D( T* d_a, const vector<T>& h_a )
{
  cudaMemcpyH2D( d_a, &h_a[0], h_a.size() );
}

template <class T>
inline void cudaMemcpyD2H( vector<T>& h_a, const T* d_a )
{
  cudaMemcpyD2H( &h_a[0], d_a, h_a.size() );
}



inline void cudaDelete( void* d_a )
{
  cudaFree( d_a );
  CHECKCUDA("cudaFree Error");
}



template <class T>
inline T* cudaNew( int N, const T* h_a = NULL )
{
  T* d_a = NULL;

  cudaMalloc( (void**)&d_a, N*sizeof(T) );
  CHECKCUDA("Malloc Error");

  if( h_a != NULL )
    cudaMemcpyH2D( d_a, h_a, N );

  return d_a;
}

template <class T>
inline T* cudaNew( const vector<T>& h_a )
{
  return cudaNew( h_a.size(), &h_a[0] );
}


inline void cudaInit(int device = 0)
{
  StopWatch initTimer;

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if( deviceCount == 0 ) {
    cout << "Error: No devices supporting CUDA" << endl;
    exit(1);
  }
  
  if( device < 0 )               device = 0;
  if( device > deviceCount - 1 ) device = deviceCount - 1;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  if (deviceProp.major < 1) {
    cerr << "Error: " << deviceProp.name << " does not support CUDA." << endl;
    exit(1);
  }

  cudaSetDevice(device);
  cerr << "Initializing " << deviceProp.name << "... ";
  
  int* temp = cudaNew<int>(1);
  cudaDelete( temp );

  cerr << initTimer.stop() << "s" << endl << endl;
}












#endif
