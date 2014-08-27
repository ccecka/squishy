#ifndef ASSEMBLYUTIL_H
#define ASSEMBLYUTIL_H

#include "../General.h"
#include "../General.cu"
#include "../MyMatrix/Matrix.h"


/* Input:
 * Raw[k] - A list of data to be read by thread block k
 *
 * Output:
 * partPtr[k] - A pointer (int) into gpuArray from which thread block k
 *              will start reading
 * gpuArray - The data to be read by the GPU in coalesced reads
 */
template <typename D>
void create_GPU_Arrays( vector< list<D> >& Raw,
			vector<int>& partPtr,
			vector<D>& gpuArray )
{
  partPtr.resize( Raw.size()+1 );
  partPtr[0] = 0;

  // Determine the size of gpuArray
  int size = 0;
  int rawSize = Raw.size();
  for( int id = 0; id < rawSize; ++id ) {
    size += round_up( Raw[id].size(), WARP_SIZE );
  }
  gpuArray.resize( size );
  // Default values are -1
  gpuArray.assign( gpuArray.size(), -1 );

  // Assemble the partPtr and gpuArray
  for( int id = 0; id < rawSize; ++id ) {

    // The data list for this block
    list<D>& dList = Raw[id];
    // The current index into gpuArray
    int nIndex = partPtr[id];

    typename list<D>::iterator li;
    for( li = dList.begin(); li != dList.end(); ++li ) {
      gpuArray[nIndex++] = *li;
    }

    // Update the partition ptr, keeping the data aligned
    partPtr[id+1] = partPtr[id] + round_up( dList.size(), WARP_SIZE );
  }
}


/* Input:
 * Raw[k] - A list of data to be read by thread block k.
 * The nth list is to be read by the nth thread with all reads coalesced.
 * All data vectors need to be the same length, otherwise use LPT
 *
 * Output:
 * partPtr[k] - A pointer (int) into gpuArray from which thread block k
 *              will start reading
 * gpuArray - The data to be read by the GPU in coalesced reads
 */
template <typename D>
void create_GPU_Arrays( vector< list< vector<D> > >& Raw,
			vector<int>& partPtr,
			vector<D>& gpuArray,
			int blockSize )
{
  partPtr.resize( Raw.size()+1 );
  partPtr[0] = 0;

  // Determine the number of columns needed
  int numColumns = 0;
  int rawSize = Raw.size();
  int vecSize = Raw[0].begin()->size();

  for( int id = 0; id < rawSize; ++id ) {
    // Determine the number of vectors that need to go back-to-back
    int stackNum = DIVIDE_INTO( Raw[id].size(), blockSize );
    // Make sure the data vector sizes are the same size
    typename list< vector<D> >::iterator lli;
    for( lli = Raw[id].begin(); lli != Raw[id].end(); ++lli ) {
      assert( vecSize == (int)lli->size() );
    }
    // Then stackNum * vecSize columns are needed
    numColumns += stackNum * vecSize;
  }

  // Use a column-major matrix with blockSize rows and 0 default value
  matrix<D,COL_MAJOR> gpuMatrix( blockSize, numColumns, 0 );

  // Assemble the partPtr and gpuArray
  int currentCol = 0;
  int currentRow = 0;
  // For each block
  for( int id = 0; id < rawSize; ++id ) {

    // The list of data vectors for this block
    list< vector<D> >& dList = Raw[id];

    typename list< vector<D> >::iterator li;
    for( li = dList.begin(); li != dList.end(); ++li ) {

      if( currentRow == blockSize ) {
	currentRow = 0;
	currentCol += vecSize;
      }

      // For each data vector, store in gpuMatrix along the currentRow
      vector<D>& dvector = *li;
      for( int k = 0; k < vecSize; ++k ) {
	gpuMatrix(currentRow, k + currentCol) = dvector[k];
      }

      ++currentRow;
    }

    // Onto the next block
    currentRow = 0;
    currentCol += vecSize;

    // Update the partition ptr
    partPtr[id+1] = currentCol * blockSize;
  }

  // Copy into gpuArray   HACKy casting... FIXME?
  gpuArray = (vector<D>&) ((vector_cpu<D>&) gpuMatrix);
}


/* Input:
 * Raw[k] - A list of data lists to be read (coalesced) by thread block k
 * blockSize - The number of threads per block
 *
 * Ouput:
 * partPtr[k] - A pointer (int) into gpuArray from which thread block k
 *              will start reading
 * gpuArray - A single array which forms a column-major matrix for each
 *            thread block where the columns are to be read in coalesced
 *            transactions
 */
template <typename D>
void create_GPU_Arrays_LPT( vector< list< list<D> > >& Raw,
			    vector<int>& partPtr,
			    vector<D>& gpuArray,
			    int blockSize )
{
  partPtr.resize( Raw.size()+1 );
  partPtr[0] = 0;

  // Determine the number of columns needed
  int numColumns = 0;
  int rawSize = Raw.size();

  for( int id = 0; id < rawSize; ++id ) {
    // Determine the number of lists need to go back-to-back
    int stackNum = DIVIDE_INTO( Raw[id].size(), blockSize );
    // Determine the longest list
    int maxSize = 0;
    typename list< list<D> >::iterator lli;
    for( lli = Raw[id].begin(); lli != Raw[id].end(); ++lli ) {
      maxSize = max( maxSize, (int) (*lli).size() );
    }
    // Then stackNum * maxSize columns may be needed
    numColumns += stackNum * maxSize;
  }

  // Use a column-major matrix with blockSize rows and 0 default value
  matrix<D,COL_MAJOR> gpuMatrix( blockSize, numColumns, 0 );

  // Initialize the bin priority queue for the LPT algorithm
  // The bins are rows sorted from least full to most full
  priority_queue< pair<int,int>,
                  deque< pair<int,int> >,
                  greater< pair<int,int> > > pq;
  // Insert each bin with 0 entries to the queue
  for( int row = 0; row < blockSize; ++row ) {
    pq.push( pair<int,int>(0,row) );
  }

  // Construct the compressed GPU arrays

  // For each block
  for( int id = 0; id < rawSize; ++id ) {
    list< list<D> >& partLists = Raw[id];

    // Sort the data lists by their cardinality
    partLists.sort(compare_size< list<D> >);

    // For each data list (largest first)
    typename list< list<D> >::reverse_iterator rli;
    for( rli = partLists.rbegin(); rli != partLists.rend(); ++rli ) {
      list<D>& dataList = *rli;

      // Insert the dataList into the smallest "bin" of the gpuMatrix
      int currentCol = pq.top().first;
      int currentRow = pq.top().second;
      // Remove this bin
      pq.pop();

      // Insert the dataList into the gpuMatrix along a row
      typename list<D>::iterator li;
      for( li = dataList.begin(); li != dataList.end(); ++li ) {
	gpuMatrix(currentRow, currentCol) = *li;
	++currentCol;
      }

      // Reinsert this bin with its new size
      pq.push( pair<int,int>(currentCol, currentRow) );
    } // end LPT data lists

    // Onto the next block

    // Get the largest bin
    while( pq.size() != 1 ) pq.pop();
    int currentCol = pq.top().first;
    pq.pop();

    // Insert each bin with currentCol entries to the queue
    for( int row = 0; row < blockSize; ++row ) {
      pq.push( pair<int,int>(currentCol, row) );
    }

    // Update the Partition Ptr array
    partPtr[id+1] = currentCol * blockSize;
  } // end each partition

  // Copy into gpuArray    HACKy casting... FIXME?
  gpuArray = (vector<D>&) ((vector_cpu<D>&) gpuMatrix);

  // Resize to the size that was actually used
  gpuArray.resize( partPtr[Raw.size()] );
}


#endif
