#ifndef ASSEMBLYUTIL_CU
#define ASSEMBLYUTIL_CU

template <int BLOCKSIZE>
__device__ void scatterXF3( int tid, int end,
			    float* coord, float* force, float* target,
			    int* scatterArray )
{
  // Can group into float3 if single-precision
  float3 nCoord, nForce;
  int n;
  while( tid < end ) {
    n = scatterArray[tid];
    if( n < 0 ) {
      nCoord = ((float3*)coord)[-n-1];
      nForce = ((float3*)force)[-n-1];
    } else if( n > 0 ) {
      ((float3*)(target + n - 1))[0] = nCoord;
      ((float3*)(target + n - 1))[1] = nForce;
    }
    tid += BLOCKSIZE;
  }
}

template <int BLOCKSIZE>
__device__ void scatterXF3( int tid, int end,
			    double* coord, double* force, double* target,
			    int* scatterArray )
{
  // Can group into float3 if single-precision
  double3 nCoord, nForce;
  int n;
  while( tid < end ) {
    n = scatterArray[tid];
    if( n < 0 ) {
      nCoord = ((double3*)coord)[-n-1];
      nForce = ((double3*)force)[-n-1];
    } else if( n > 0 ) {
      ((double3*)(target + n - 1))[0] = nCoord;
      ((double3*)(target + n - 1))[1] = nForce;
    }
    tid += BLOCKSIZE;
  }
}


template <int BLOCKSIZE, typename T>
__device__ void reduce( int tid, int end,
			T* source, T* target,
			int* reduceArray )
{
  // Assemble all element data by NZ
  T kf = 0;
  int n;
  // Shift pointers to prevent source[n-1] and target[n-1]
  source = source - 1;
  target = target - 1;

  while( tid < end ) {
    n = reduceArray[tid];
    if( n > 0 ) { kf += source[n]; }
    if( n < 0 ) { target[-n] = kf; kf = 0; }
    tid += BLOCKSIZE;
  }
}

template <int BLOCKSIZE, typename T>
__device__ void cuda_copy( int tid, int end, T* source, T* target )
{
  while( tid < end ) {
    target[tid] = source[tid];
    tid += BLOCKSIZE;
  }
}



#endif
