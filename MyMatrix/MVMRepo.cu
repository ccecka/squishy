#ifndef MVMREPO_CU
#define MVMREPO_CU


/************************************/
/********** General Support *********/
/************************************/

// These textures are (optionally) used to cache the 'x' vector in y += A*x
texture<float,1> tex_x_float;
texture<int2,1>  tex_x_double;

// Float Version (Double requires int2 texture)
template <bool USE_TEX>
__inline__ __device__ float fetch_x( const int& i, const float* x )
{
  if( USE_TEX )
    return tex1Dfetch(tex_x_float, i);
  else
    return x[i];
}

template <bool USE_TEX>
__inline__ __device__ double fetch_x( const int& i, const double* x )
{
  if( USE_TEX ) {
    int2 v = tex1Dfetch(tex_x_double, i);
    return __hiloint2double(v.y, v.x);
  } else
    return x[i];
}

template <bool USE_TEX>
inline void bind_x( const float* x )
{   
  if( USE_TEX ) {
    size_t offset = size_t(-1);
    cudaBindTexture(&offset, tex_x_float, x);
    if( offset != 0 ) {
      cerr << "memory is not aligned, refusing to use texture cache" << endl;
      exit(1);
    }
  }
}

// Use int2 to pull doubles through texture cache
template <bool USE_TEX>
inline void bind_x( const double* x )
{   
  if( USE_TEX ) {
    size_t offset = size_t(-1);
    cudaBindTexture(&offset, tex_x_double, x);
    if( offset != 0 ) {
      cerr << "memory is not aligned, refusing to use texture cache" << endl;
      exit(1);
    }
  }
}

// segmented reduction in shared memory
template <typename T>
__device__ T segreduce_warp(const int lane, int row, 
			    T val, int* rows, T* vals)
{
  const int tid = threadIdx.x;
  
  rows[tid] = row;
  vals[tid] = val;
  
  if(lane >=  1 && row == rows[tid- 1]) vals[tid] = val = val + vals[tid- 1];
  if(lane >=  2 && row == rows[tid- 2]) vals[tid] = val = val + vals[tid- 2];
  if(lane >=  4 && row == rows[tid- 4]) vals[tid] = val = val + vals[tid- 4];
  if(lane >=  8 && row == rows[tid- 8]) vals[tid] = val = val + vals[tid- 8];
  if(lane >= 16 && row == rows[tid-16]) vals[tid] = val = val + vals[tid-16];
  
  return val;
}

template <typename T>
__device__ void segreduce_block(const int* idx, T* val)
{
  const int tid = threadIdx.x;
  T left = 0;

  if(tid >=   1 && idx[tid] == idx[tid-  1] ) { left = val[tid-  1]; } 
  __syncthreads(); val[tid] += left; left = 0; __syncthreads();  
  if(tid >=   2 && idx[tid] == idx[tid-  2] ) { left = val[tid-  2]; } 
  __syncthreads(); val[tid] += left; left = 0; __syncthreads();
  if(tid >=   4 && idx[tid] == idx[tid-  4] ) { left = val[tid-  4]; } 
  __syncthreads(); val[tid] += left; left = 0; __syncthreads();
  if(tid >=   8 && idx[tid] == idx[tid-  8] ) { left = val[tid-  8]; } 
  __syncthreads(); val[tid] += left; left = 0; __syncthreads();
  if(tid >=  16 && idx[tid] == idx[tid- 16] ) { left = val[tid- 16]; } 
  __syncthreads(); val[tid] += left; left = 0; __syncthreads();
  if(tid >=  32 && idx[tid] == idx[tid- 32] ) { left = val[tid- 32]; } 
  __syncthreads(); val[tid] += left; left = 0; __syncthreads();  
  if(tid >=  64 && idx[tid] == idx[tid- 64] ) { left = val[tid- 64]; } 
  __syncthreads(); val[tid] += left; left = 0; __syncthreads();
  if(tid >= 128 && idx[tid] == idx[tid-128] ) { left = val[tid-128]; } 
  __syncthreads(); val[tid] += left; left = 0; __syncthreads();
  if(tid >= 256 && idx[tid] == idx[tid-256] ) { left = val[tid-256]; } 
  __syncthreads(); val[tid] += left; left = 0; __syncthreads();
}



/************************************/
/********** COO_Matrix **************/
/************************************/

template <unsigned int BLOCK_SIZE, bool USE_TEX, typename T>
__global__ void coo_mvm_flat(const int num_nonzeros,
			     const int interval_size,
			     const int* I, 
			     const int* J, 
			     const T* V, 
			     const T* x, 
			     T* y,
			     int* temp_rows,
			     T* temp_vals)
{
  __shared__ volatile int rows[48 *(BLOCK_SIZE/WARP_SIZE)];
  __shared__ volatile T vals[BLOCK_SIZE];

  const int tid = threadIdx.x;
  const int thread_id   = BLOCK_SIZE * blockIdx.x + tid;
  const int thread_lane = tid & (WARP_SIZE-1);
  const int warp_id     = thread_id   / WARP_SIZE;  

  const int interval_begin = warp_id * interval_size;
  const int interval_end   = min(interval_begin + interval_size, num_nonzeros);

  const int idx = (WARP_SIZE/2) * (tid/WARP_SIZE + 1) + tid;

  rows[idx - 16] = -1;

  if( interval_begin >= interval_end )
    return;

  if( thread_lane == 31 ) {
    // initialize the carry in values
    rows[idx] = I[interval_begin]; 
    vals[tid] = 0;
  }
  
  for(int n = interval_begin + thread_lane; n < interval_end; n += WARP_SIZE) {
    int row = I[n];                                // row index (i)
    T val = V[n] * fetch_x<USE_TEX>(J[n], x);      // A(i,j) * x(j)
        
    if( thread_lane == 0 ) {
      if(row == rows[idx + 31])
	val += vals[tid + 31];                         // row continues
      else
	y[rows[idx + 31]] += vals[tid + 31];           // row terminated
    }
    
    rows[idx]         = row;
    vals[tid] = val;
    
    if(row == rows[idx -  1]) { vals[tid] = val = val + vals[tid -  1]; } 
    if(row == rows[idx -  2]) { vals[tid] = val = val + vals[tid -  2]; }
    if(row == rows[idx -  4]) { vals[tid] = val = val + vals[tid -  4]; }
    if(row == rows[idx -  8]) { vals[tid] = val = val + vals[tid -  8]; }
    if(row == rows[idx - 16]) { vals[tid] = val = val + vals[tid - 16]; }
    
    if( thread_lane < 31 && row != rows[idx + 1] )
      y[row] += vals[tid];                             // row terminated
  }

  if( thread_lane == 31 ) {
    // write the carry out values
    temp_rows[warp_id] = rows[idx];
    temp_vals[warp_id] = vals[tid];
  }
}

// The second level of the segmented reduction operation
template <unsigned int BLOCK_SIZE, typename T>
__global__ void coo_mvm_reduce(const int num_warps,
			       const int* temp_rows,
			       const T* temp_vals,
			       T* y)
{
  __shared__ int rows[BLOCK_SIZE + 1];    
  __shared__ T vals[BLOCK_SIZE + 1];    

  const int tid = threadIdx.x;
  const int end = num_warps - (num_warps & (BLOCK_SIZE - 1));

  if( tid == 0 ) {
    rows[BLOCK_SIZE] = (int) -1;
    vals[BLOCK_SIZE] = (T) 0;
  }
    
  __syncthreads();
  
  int i = tid;
  
  while( i < end ) {
    // do full blocks
    rows[tid] = temp_rows[i];
    vals[tid] = temp_vals[i];
    
    __syncthreads();
    
    segreduce_block(rows, vals);
    
    if( rows[tid] != rows[tid + 1] )
      y[rows[tid]] += vals[tid];
    
    __syncthreads();
    
    i += BLOCK_SIZE; 
  }
  
  if( end < num_warps ) {
    if( i < num_warps ){
      rows[tid] = temp_rows[i];
      vals[tid] = temp_vals[i];
    } else {
      rows[tid] = (int) -1;
      vals[tid] = (T)  0;
    }
    
    __syncthreads();
    
    segreduce_block(rows, vals);
    
    if( i < num_warps )
      if( rows[tid] != rows[tid + 1] )
	y[rows[tid]] += vals[tid];
  }
}

template <typename T>
__global__ void coo_mvm_serial(const int num_nonzeros,
			       const int* I, const int* J, 
			       const T* A, const T* x, T* y)
{
  for( int n = 0; n < num_nonzeros; ++n ) {
    y[I[n]] += A[n] * x[J[n]];
  }
}


/************************************/
/********** CSR_Matrix **************/
/************************************/


// Compute y = Ax     CSR_scalar
template <unsigned int BLOCK_SIZE, bool USE_TEX, typename T>
__global__ void csr_mvm_scalar( int N, int* rowptr, int* colidx, 
				T* A, const T* x, T* y )
{
  int t_row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  
  if( t_row < N ) {
    int row_begin = rowptr[t_row];
    int row_end   = rowptr[t_row+1];

    T sum = 0;
    for( int col = row_begin; col < row_end; ++col ) {
      sum += A[col] * fetch_x<USE_TEX>(colidx[col], x);
    }

    y[t_row] = sum;
  }
}


// Compute y = Ax     DCSR_scalar
template <unsigned int BLOCK_SIZE, bool USE_TEX, typename T>
__global__ void dcsr_mvm_scalar( int N, int* rowptr, int* colidx, 
				 T* A, const T* x, T* y )
{
  int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  
  if( row < N ) {
    int row_begin = rowptr[row];
    int row_end   = rowptr[row+1];

    T sum = A[row] * x[row];
    for( int col = row_begin; col < row_end; ++col ) {
      sum += A[col] * fetch_x<USE_TEX>(colidx[col], x);
    }

    y[row] = sum;
  }
}


// Compute y = Ax     CSR_vector
template <unsigned int VEC_PER_BLOCK, unsigned int THR_PER_VEC, bool USE_TEX,
	  typename T>
__global__ void csr_mvm_vector( int num_rows, int* rowptr, int* colidx, 
			        T* A, const T* x, T* y )
{
  __shared__ volatile T sM[VEC_PER_BLOCK * THR_PER_VEC + THR_PER_VEC / 2];
  __shared__ volatile int ptrs[VEC_PER_BLOCK][2];
    
  const int THR_PER_BLOCK = VEC_PER_BLOCK * THR_PER_VEC;

  const int tidx        = threadIdx.x;
  const int thread_id   = THR_PER_BLOCK * blockIdx.x + tidx;    
  const int thread_lane = tidx & (THR_PER_VEC - 1);
  const int vector_id   = thread_id   /  THR_PER_VEC;
  const int vector_lane = tidx /  THR_PER_VEC;
  const int num_vectors = VEC_PER_BLOCK * gridDim.x;

  for( int row = vector_id; row < num_rows; row += num_vectors ) {
    // use two threads to fetch Ap[row] and Ap[row+1]
    // this is considerably faster than the straightforward version
    if(thread_lane < 2)
      ptrs[vector_lane][thread_lane] = rowptr[row + thread_lane];
    
    const int row_start = ptrs[vector_lane][0];    // Ap[row];
    const int row_end   = ptrs[vector_lane][1];    // Ap[row+1];
    
    // initialize local sum
    T sum = 0;
    
    if( THR_PER_VEC == 32 && row_end - row_start > 32 ) {
      // ensure aligned memory access to Aj and Ax
      int jj = row_start - (row_start & (THR_PER_VEC-1)) + thread_lane;

      // accumulate local sums
      if( jj >= row_start && jj < row_end )
	sum += A[jj] * fetch_x<USE_TEX>(colidx[jj], x);
      
      // accumulate local sums
      for( jj += THR_PER_VEC; jj < row_end; jj += THR_PER_VEC )
	sum += A[jj] * fetch_x<USE_TEX>(colidx[jj], x);
    } else {
      // accumulate local sums
      for( int jj = row_start + thread_lane; jj < row_end; jj += THR_PER_VEC )
	sum += A[jj] * fetch_x<USE_TEX>(colidx[jj], x);
    }
    
    // store local sum in shared memory
    sM[threadIdx.x] = sum;
    
    // reduce local sums to row sum
    if(THR_PER_VEC > 16) sM[tidx] = sum = sum + sM[tidx + 16];
    if(THR_PER_VEC >  8) sM[tidx] = sum = sum + sM[tidx +  8];
    if(THR_PER_VEC >  4) sM[tidx] = sum = sum + sM[tidx +  4];
    if(THR_PER_VEC >  2) sM[tidx] = sum = sum + sM[tidx +  2];
    if(THR_PER_VEC >  1) sM[tidx] = sum = sum + sM[tidx +  1];
    
    // first thread writes the result
    if( thread_lane == 0 )
      y[row] = sM[tidx];
  }
}

// Compute y = Ax     CSR_vector
template <unsigned int VEC_PER_BLOCK, unsigned int THR_PER_VEC, bool USE_TEX,
	  typename T>
__global__ void dcsr_mvm_vector( int num_rows, int* rowptr, int* colidx, 
				 T* A, const T* x, T* y )
{
  __shared__ volatile T sM[VEC_PER_BLOCK * THR_PER_VEC + THR_PER_VEC / 2];
  __shared__ volatile int ptrs[VEC_PER_BLOCK][2];
    
  const int THR_PER_BLOCK = VEC_PER_BLOCK * THR_PER_VEC;

  const int tidx        = threadIdx.x;
  const int thread_id   = THR_PER_BLOCK * blockIdx.x + tidx;    
  const int thread_lane = tidx & (THR_PER_VEC - 1);
  const int vector_id   = thread_id   /  THR_PER_VEC;
  const int vector_lane = tidx /  THR_PER_VEC;
  const int num_vectors = VEC_PER_BLOCK * gridDim.x;

  for( int row = vector_id; row < num_rows; row += num_vectors ) {
    // use two threads to fetch Ap[row] and Ap[row+1]
    // this is considerably faster than the straightforward version
    if(thread_lane < 2)
      ptrs[vector_lane][thread_lane] = rowptr[row + thread_lane];
    
    const int row_start = ptrs[vector_lane][0];    // Ap[row];
    const int row_end   = ptrs[vector_lane][1];    // Ap[row+1];
    
    // initialize local sum
    T sum = 0;
    
    if( THR_PER_VEC == 32 && row_end - row_start > 32 ) {
      // ensure aligned memory access to Aj and Ax
      int jj = row_start - (row_start & (THR_PER_VEC-1)) + thread_lane;

      // accumulate local sums
      if( jj >= row_start && jj < row_end )
	sum += A[jj] * fetch_x<USE_TEX>(colidx[jj], x);
      
      // accumulate local sums
      for( jj += THR_PER_VEC; jj < row_end; jj += THR_PER_VEC )
	sum += A[jj] * fetch_x<USE_TEX>(colidx[jj], x);
    } else {
      // accumulate local sums
      for( int jj = row_start + thread_lane; jj < row_end; jj += THR_PER_VEC )
	sum += A[jj] * fetch_x<USE_TEX>(colidx[jj], x);
    }
    
    // store local sum in shared memory
    sM[threadIdx.x] = sum;
    
    // reduce local sums to row sum
    if(THR_PER_VEC > 16) sM[tidx] = sum = sum + sM[tidx + 16];
    if(THR_PER_VEC >  8) sM[tidx] = sum = sum + sM[tidx +  8];
    if(THR_PER_VEC >  4) sM[tidx] = sum = sum + sM[tidx +  4];
    if(THR_PER_VEC >  2) sM[tidx] = sum = sum + sM[tidx +  2];
    if(THR_PER_VEC >  1) sM[tidx] = sum = sum + sM[tidx +  1];
    
    // first thread writes the result
    if( thread_lane == 0 )
      y[row] = sM[tidx] + A[row] * fetch_x<USE_TEX>(row, x);;
  }
}



/************************************/
/********** ELL_Matrix **************/
/************************************/


// Compute y = Ax     HYB_MVM
template <unsigned int BLOCK_SIZE, bool USE_TEX, typename T>
__global__ void ell_mvm(int n_rows, int ell_rows, int ell_size,
			int* ellidx, T* A, const T* x, T* y)
{
  const int thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  const int grid_size = gridDim.x * BLOCK_SIZE;

  for( int row = thread_id; row < n_rows; row += grid_size ) {
    T sum = 0;
    for( int offset = row; offset < ell_size; offset += ell_rows ) {
      const int col = ellidx[offset];
      if( col != -1 ) {       // matrix_hyb<T>::INVALID_INDEX ) {
	sum += A[offset] * fetch_x<USE_TEX>(col, x);
      }
    }
    y[row] = sum;
  }
}


// Compute y = Ax     HYB_MVM
template <unsigned int BLOCK_SIZE, bool USE_TEX, typename T>
__global__ void dell_mvm(int DNZ, int n_rows, int ell_rows, int ell_size,
			 int* ellidx, T* A, const T* x, T* y)
{
  const int thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  const int grid_size = gridDim.x * BLOCK_SIZE;

  for( int row = thread_id; row < n_rows; row += grid_size ) {
    T sum = A[row] * x[row];
    for( int offset = row; offset < ell_size; offset += ell_rows ) {
      const int col = ellidx[offset];
      if( col != -1 ) {       // matrix_hyb<T>::INVALID_INDEX ) {
	sum += A[DNZ+offset] * fetch_x<USE_TEX>(col, x);
      }
    }
    y[row] = sum;
  }
}

#endif
