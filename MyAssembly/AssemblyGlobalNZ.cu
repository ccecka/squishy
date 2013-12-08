#ifndef ASSEMBLYGLOBALNZCUDA_CU
#define ASSEMBLYGLOBALNZCUDA_CU

#include "../General.cu"
#include "AssemblyUtil.cu"

#include "../Problem.h"

#define NDOF 3   // Node degrees of freedom
#define NPE  4   // Nodes per element
#define VNPE 4   // Vertex nodes per element
#define NDIM 3   // Number of dimensions
#define EDOF (NPE*NDOF)   // Element degrees of freedom


template <int TYPE, int BLOCK_SIZE, typename T>
__global__ void computeElems( T* E,
			      T* coord, T* force,
			      int* nPartPtr, int* nodeArray,
			      int* eIENPartPtr, T* eIENArray )
{
  // Prefetch partition pointers
  extern __shared__ T sMem[];
  __shared__ int sPtr[4];
  int tid = threadIdx.x;
  
  if( tid <= 1 ) {
    sPtr[tid]   = nPartPtr[blockIdx.x + tid];
    sPtr[2+tid] = eIENPartPtr[blockIdx.x + tid];
  }

  __syncthreads();

  // Prefetch nodal data to shared memory
  // Alot of this could be coalesced if we renumbered the nodes correctly
  nodeArray += sPtr[0];
  int end = sPtr[1] - sPtr[0];

  while( tid < end ) {
    int n = nodeArray[tid];
    if( n != -1 ) {
      if( sizeof(T) == 4 ) {
	// float3
	((float3*)sMem)[2*tid]   = ((float3*)coord)[n];	
	((float3*)sMem)[2*tid+1] = ((float3*)force)[n];
      } else if( sizeof(T) == 8 ) {
	// double3
	((double3*)sMem)[2*tid]   = ((double3*)coord)[n];	
	((double3*)sMem)[2*tid+1] = ((double3*)force)[n];
      }
    }
    tid += BLOCK_SIZE;
  }

  __syncthreads();

  // Compute the element data and store in global mem
  E += (sPtr[2]/(VNPE+6)) * ((EDOF*(EDOF+3))/2) + threadIdx.x;
  tid = sPtr[2] + threadIdx.x;
  end = sPtr[3];
  T* sTemp;

  while( tid < end ) {
    /*
    // Optimized version?
    T nodes1 = eIENArray[tid];   tid += blockDim.x;
    T nodes2 = eIENArray[tid];   tid += blockDim.x;

    unsigned short n1 = (reinterpret_cast<ushort2*>(&nodes1))->y;
    unsigned short n2 = (reinterpret_cast<ushort2*>(&nodes1))->x;
    unsigned short n3 = (reinterpret_cast<ushort2*>(&nodes2))->y;
    unsigned short n4 = (reinterpret_cast<ushort2*>(&nodes2))->x;
    */
    
    sTemp = sMem + 6 * (int) eIENArray[tid];   tid += BLOCK_SIZE;
    const T  x1 = sTemp[0],  y1 = sTemp[1],  z1 = sTemp[2];
    const T bx1 = sTemp[3], by1 = sTemp[4], bz1 = sTemp[5];
    sTemp = sMem + 6 * (int) eIENArray[tid];   tid += BLOCK_SIZE;
    const T  x2 = sTemp[0],  y2 = sTemp[1],  z2 = sTemp[2];
    const T bx2 = sTemp[3], by2 = sTemp[4], bz2 = sTemp[5];
    sTemp = sMem + 6 * (int) eIENArray[tid];   tid += BLOCK_SIZE;
    const T  x3 = sTemp[0],  y3 = sTemp[1],  z3 = sTemp[2];
    const T bx3 = sTemp[3], by3 = sTemp[4], bz3 = sTemp[5];
    sTemp = sMem + 6 * (int) eIENArray[tid];   tid += BLOCK_SIZE;
    const T  x4 = sTemp[0],  y4 = sTemp[1],  z4 = sTemp[2];
    const T bx4 = sTemp[3], by4 = sTemp[4], bz4 = sTemp[5];

    const T Jinv11 = eIENArray[tid];   tid += BLOCK_SIZE;
    const T Jinv12 = eIENArray[tid];   tid += BLOCK_SIZE;
    const T Jinv13 = eIENArray[tid];   tid += BLOCK_SIZE;
    const T Jinv22 = eIENArray[tid];   tid += BLOCK_SIZE;
    const T Jinv23 = eIENArray[tid];   tid += BLOCK_SIZE;
    const T Jinv33 = eIENArray[tid];   tid += BLOCK_SIZE;

    if( Jinv33 != 0 ) {
      Problem<T>::Tetrahedral<TYPE,BLOCK_SIZE>( x1, y1, z1, bx1, by1, bz1,
						x2, y2, z2, bx2, by2, bz2,
						x3, y3, z3, bx3, by3, bz3,
						x4, y4, z4, bx4, by4, bz4,
						Jinv11, Jinv12, Jinv13, 
						Jinv22, Jinv23, 
						Jinv33,
						E );
      
      E += BLOCK_SIZE * ((EDOF*(EDOF+3))/2);
    }
  }
}


template <int BLOCK_SIZE, typename T>
__global__ void assembleGlobalNZ( T* E, T* KF,
				  int* redPartPtr, int* redList, 
				  int nzPart, int nzTot )
{
  extern __shared__ T sMem[];
  __shared__ int sPtr[2];
  int tid = threadIdx.x;

  // Prefetch the block pointers
  if( tid <= 1 ) {
    sPtr[tid] = redPartPtr[blockIdx.x+tid];
  }
  
  __syncthreads();

  // Assemble all element data by NZ from E into sMem
  reduce<BLOCK_SIZE>( tid + sPtr[0], sPtr[1], E, sMem, redList );

  __syncthreads();

  // Copy the sMem into KF with a coalesced push
  cuda_copy<BLOCK_SIZE>( tid, (int) min( nzPart, nzTot - nzPart*blockIdx.x ), 
			 sMem, KF + nzPart * blockIdx.x );
}


#endif
