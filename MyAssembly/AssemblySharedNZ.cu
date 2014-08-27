#ifndef ASSEMBLYSHAREDNZCUDA_CU
#define ASSEMBLYSHAREDNZCUDA_CU

#include "../General.cu"

#include "../Problem.h"

#define NDOF 3   // Node degrees of freedom
#define NPE  4   // Nodes per element
#define VNPE 4   // Corner nodes per element
#define NDIM 3   // Number of dimensions
#define EDOF (NPE*NDOF)   // Element degrees of freedom


template <int TYPE, int BLOCK_SIZE, typename T>
__global__ void assembleSharedNZ( T* coord, T* force,
				  int* scatterPartPtr, int* scatterArray,
				  int* eNumPart,
				  int* suppPtr, T* suppData,
				  int* redPartPtr, int* redList,
				  T* KF )
{
  // Prefetch partition pointers
  extern __shared__ T sMem[];
  __shared__ int sPtr[6];
  int tid = threadIdx.x;

  // Prefetch pointers
  if( tid <= 1 ) {
    sPtr[tid] = scatterPartPtr[blockIdx.x + tid];
    sPtr[2+tid] = redPartPtr[blockIdx.x + tid];
  }
  if( tid == 2 ) {
    sPtr[4] = eNumPart[blockIdx.x];
  }
  if( tid == 3 ) {
    sPtr[5] = suppPtr[blockIdx.x];
  }

  __syncthreads();

  // Scatter nodal data to shared memory

  // Prefetch element coord and BC data
  scatterXF3<BLOCK_SIZE>( tid + sPtr[0], sPtr[1],
			  coord, force, sMem,
			  scatterArray );

  __syncthreads();

  // Compute the element data and store in shared mem
  int numE = sPtr[4];
  T* sE = sMem + tid * ((EDOF*(EDOF+3))/2);
  suppData += sPtr[5] + tid;

  while( tid < numE ) {

    const T  x1 = sE[ 0],  y1 = sE[ 1],  z1 = sE[ 2];
    const T bx1 = sE[ 3], by1 = sE[ 4], bz1 = sE[ 5];
    const T  x2 = sE[ 6],  y2 = sE[ 7],  z2 = sE[ 8];
    const T bx2 = sE[ 9], by2 = sE[10], bz2 = sE[11];
    const T  x3 = sE[12],  y3 = sE[13],  z3 = sE[14];
    const T bx3 = sE[15], by3 = sE[16], bz3 = sE[17];
    const T  x4 = sE[18],  y4 = sE[19],  z4 = sE[20];
    const T bx4 = sE[21], by4 = sE[22], bz4 = sE[23];

    const T Jinv11 = suppData[0*BLOCK_SIZE];
    const T Jinv12 = suppData[1*BLOCK_SIZE];
    const T Jinv13 = suppData[2*BLOCK_SIZE];
    const T Jinv22 = suppData[3*BLOCK_SIZE];
    const T Jinv23 = suppData[4*BLOCK_SIZE];
    const T Jinv33 = suppData[5*BLOCK_SIZE];

    Problem<T>::Tetrahedral<TYPE,1>( x1, y1, z1, bx1, by1, bz1,
				     x2, y2, z2, bx2, by2, bz2,
				     x3, y3, z3, bx3, by3, bz3,
				     x4, y4, z4, bx4, by4, bz4,
				     Jinv11, Jinv12, Jinv13,
				     Jinv22, Jinv23,
				     Jinv33,
				     sE );

    tid += BLOCK_SIZE;
    sE += BLOCK_SIZE * ((EDOF*(EDOF+3))/2);
    suppData += BLOCK_SIZE * 6;
  }

  __syncthreads();

  // Assemble
  reduce<BLOCK_SIZE>(threadIdx.x + sPtr[2], sPtr[3],
		     sMem, KF,
		     redList );
}


#endif
