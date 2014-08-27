#ifndef ASSEMBLYGLOBALNZ_H
#define ASSEMBLYGLOBALNZ_H

#include "Assembly_Interface.h"

#include "../Parser.h"

#include "AssemblyUtil.h"
#include "AssemblyGlobalNZ.cu"

#include "elemRepo/kernelMapF_e.ker"
#include "elemRepo/kernelMapK_e.ker"


template <typename T>
class AssemblyGlobalNZ : public AssemblyGPU<T>
{
  using Assembly_Interface<T>::prob;
  using Assembly_Interface<T>::LM;

  using Assembly_Interface<T>::M_;
  using Assembly_Interface<T>::K_;
  using Assembly_Interface<T>::F_;

 public:

  using AssemblyGPU<T>::d_M;    // Device pointer to (diag) matrix M
  using AssemblyGPU<T>::d_K;    // Device pointer to matrix K
  using AssemblyGPU<T>::d_F;    // Device pointer to vector F
  vector_gpu<T> d_KF;           // Device pointer to matrix K and vector F

  // Elem Computation
  int nParts;                 // Number of partitions
  const static int blockSize = 256;     // CUDA blockSize
  int sMemBytes;              // CUDA shared Memory

  //vector<int> nPartPtr;       // Partition pointer into nodal array
  //vector<int> nodeArray;      // Partition nodal array
  //vector<int> eIENPartPtr;    // Partition pointer into local IEN array
  //vector<T> eIENArray;        // Partition local IEN array

  vector_gpu<T> d_E;          // Device pointer to element data
  vector_gpu<int> d_nPartPtr;            // Device pointer to nPartPtr
  vector_gpu<int> d_nodeArray;           // Device pointer to nPart
  vector_gpu<int> d_eIENPartPtr;         // Device pointer to eIENPartPtr
  vector_gpu<T> d_eIENArray;          // Device pointer to eIEN

  // Assembly
  int sMemRed;                // Shared memory for reduction
  const static int blockSizeRed = 512;   // CUDA reduction blocksize
  int nzPart;                 // Size of NZ partition

  // KF assembly
  int nzTotKF;                // The total number of nz to assemble
  int nzPartKF;               // Size of NZ partition
  int nPartsRedKF;            // Number of NZ partitions

  //vector<int> redPartPtrKF;   // Partition pointer into reduction list
  //vector<int> redArrayKF;     // Reduction list
  vector_gpu<int> d_redPartPtrKF;        // Device pointer to redPartPtr
  vector_gpu<int> d_redArrayKF;          // Device pointer to redList

  // F assembly
  int nzTotF;                 // The total number of nz to assemble
  int nPartsRedF;             // Number of NZ partitions

  //vector<int> redPartPtrF;   // Partition pointer into reduction list
  //vector<int> redArrayF;     // Reduction list
  vector_gpu<int> d_redPartPtrF;        // Device pointer to redPartPtr
  vector_gpu<int> d_redArrayF;          // Device pointer to redList

 public:

  double cuda_time1;
  double cuda_time2;


  // Everything that we can do as a precomputation
  template <template <typename> class MATRIX>
  AssemblyGlobalNZ( Problem<T>& p, MATRIX<T>& FEM_Matrix )
      : AssemblyGPU<T>( p, FEM_Matrix )
  {
    // Get Local versions for easy
    Mesh<T>& mesh = prob.getMesh();
    const matrix<int>& IEN = mesh.getIEN();
    int nEq = K_.nCols();
    int nNZ = K_.size();
    int nElems = mesh.nElems();
    int nDoF = prob.nNDOF();
    int eDoF = prob.nEDOF();
    //int nNPE = mesh.nNodesPerElem();
    int vNPE = mesh.nVertexNodesPerElem();
    //int nDim = mesh.nDim();
    int nNodes = mesh.nNodes();

    sMemBytes = cudaMaxSMEM() - 100;

    // Load or compute the element partition mesh.epart
    string fileStr = "data/GlobalNZ_" + toString(nNodes)
        + "_" + toString(sizeof(T)) + ".part";
    const char* filename = fileStr.c_str();
    if( file_exists(filename) ) {
      cout << "Reading Partition From " << fileStr << endl;
      parseBIN( filename, mesh.epart );
    } else {
      findPartition();
      writeBIN( mesh.epart, filename );
    }

    nParts = max( mesh.epart ) + 1;
    COUT_VAR( nParts );

    // Get the elements and nodes needed by each partition
    vector< list<int> > nPartList( nParts );
    vector< list<int> > ePartList( nParts );
    // For all the elements
    for( int e = 0; e < nElems; ++e ) {

      // Get the partition of the element
      int id = mesh.epart[e];

      // List elements by partition
      ePartList[id].push_back( e );

      // List triangular nodes by partition
      list<int>& nList = nPartList[id];
      for( int a = 0; a < vNPE; ++a ) {
        int n = IEN(e,a);
        nList.push_back( n );
      }
    }
    // Uniquify the nodes needed by each partition
    for( int id = 0; id < nParts; ++id ) {
      nPartList[id].sort();
      nPartList[id].unique();
    }

    // Get statistics on this partitioning
    int totE = 0;
    int maxE = 0;
    int minE = 1000000;
    int totN = 0;
    int maxN = 0;
    int minN = 1000000;
    // For each partition
    for( int id = 0; id < nParts; ++id ) {
      int nE = ePartList[id].size();
      totE += nE;
      maxE = max( maxE, nE );
      minE = min( minE, nE );

      int nN = nPartList[id].size();
      totN += nN;
      maxN = max( maxN, nN );
      minN = min( minN, nN );
    }

    cout<<"MinE: "<<minE<<"  MaxE: "<<maxE<<"  TotE: "<<totE<<endl;
    cout<<"MinN: "<<minN<<"  MaxN: "<<maxN<<"  TotN: "<<totN<<endl;

    // Make sure this will fit in sMem
    assert( maxN * prob.nodalDataSize() < sMemBytes );

    // Create elemental and nodal partition maps
    vector< map<int,int> > n2lMapPart( nParts );
    vector<int> e2lMap( nElems );
    for( int id = 0; id < nParts; ++id ) {
      list<int>& eList = ePartList[id];
      // Create a global element to partition element map
      int elocalIndex = 0;
      list<int>::iterator li;
      for( li = eList.begin(); li != eList.end(); ++li ) {
        e2lMap[*li] = elocalIndex;
        ++elocalIndex;
      }

      list<int>& nList = nPartList[id];
      // Create a global node to partition node map
      map<int,int>& n2lMap = n2lMapPart[id];
      int nlocalIndex = 0;
      for( li = nList.begin(); li != nList.end(); ++li ) {
        n2lMap[*li] = nlocalIndex;
        ++nlocalIndex;
      }
    }

    // Create coalseced GPU Arrays for reading nodal data to smem
    vector<int> nPartPtr;
    vector<int> nodeArray;

    create_GPU_Arrays( nPartList, nPartPtr, nodeArray );

    d_nPartPtr = nPartPtr;
    d_nodeArray = nodeArray;

    //cout << nPartPtr << endl;
    //cout << nodeArray << endl;

    int SIZE_EIEN = vNPE + prob.suppDataSize() / sizeof(T);

    // Create IEN lists for each partition
    vector< list< vector<T> > > IENListPart( nParts );
    // For each partition
    for( int id = 0; id < nParts; ++id ) {

      // Get the global node to partition node map
      map<int,int>& n2lMap = n2lMapPart[id];

      // For each element this partition is responsible for
      list<int>& eList = ePartList[id];
      for( list<int>::iterator li = eList.begin(); li != eList.end(); ++li ) {
        int e = *li;

        // Construct the list of nodes (in shared memory) needed by this elem
        vector<T> eIEN( SIZE_EIEN );
        for( int a = 0; a < vNPE; ++a ) {
          int n = IEN(e,a);
          int pn = n2lMap[n];

          // Add the partition node to eIEN
          eIEN[a] = pn;
        }

        // Append the supplemental data for this element
        vector<T> suppData = prob.getSuppData( e );
        for( int k = 0; k < suppData.size(); ++k ) {
          eIEN[vNPE+k] = suppData[k];
        }

        // Add eIEN to the IENList of this partition
        IENListPart[id].push_back( eIEN );
      }
    }


    // Create coalesced GPU Arrays for partition local IEN lists
    vector<int> eIENPartPtr;    // Partition pointer into local IEN array
    vector<T> eIENArray;        // Partition local IEN array

    create_GPU_Arrays(IENListPart, eIENPartPtr, eIENArray, blockSize);

    d_eIENPartPtr =  eIENPartPtr;
    d_eIENArray = eIENArray;

    //cout << mesh.epart << endl;
    //cout << mesh.IEN << endl << endl;
    //cout << "nParts: " << nParts << "      block: " << block << endl << endl;
    //cout << eIENPartPtr << endl << endl;
    //cout << eIENArray << endl << endl;


    // Construct the reduction array for each NZ of the system
    nzTotKF = nNZ + nEq;
    nzTotF = nEq;
    vector< list<int> > redList( nzTotKF );

    // For all the elements
    for( int e = 0; e < nElems; ++e ) {

      // The partition number of this element
      int id = mesh.epart[e];
      // The local partition element number
      int le = e2lMap[e];

      // Get the location in global memory that this element data begins

      // Symmetric Element Data position in global memory
      int gMemStart = (eIENPartPtr[id]/SIZE_EIEN)*((eDoF*(eDoF+3))/2)
          + (le/blockSize)*blockSize*((eDoF*(eDoF+3))/2) + (le % blockSize);

      // For each row of the element data
      for( int k1 = 0; k1 < eDoF; ++k1 ) {
        int eq1 = LM(e,k1);
        if( eq1 == -1 ) continue;

        // For each column of the element data
        for( int k2 = 0; k2 < eDoF; ++k2 ) {
          int eq2 = LM(e,k2);
          if( eq2 == -1 ) continue;

          // If symmetric and only assembilng half the matrix
          //if( eq2 <= eq1 ) continue;

          // Get the NZ of K that this element will be accumulated into
          int nz = nEq + K_.IJtoK( eq1, eq2 );

          // Get the location in global memory of this element data entry
          int data = gMemStart + blockSize * kernelMapK_e( k1, k2 );

          redList[nz].push_back( data+1 );
        }

        // The F[eq] of the forcing vector for this element
        int nz = eq1;

        // Get the location in global memory of this data entry
        int data = gMemStart + blockSize * kernelMapF_e( k1 );

        redList[nz].push_back( data+1 );
      }
    }

    // Partition the NZs for the reduction step (TODO: MATRIX DEPENDENT!)
    sMemRed = sMemBytes; // / 2.1;
    nzPart = round_down( sMemRed/sizeof(T), WARP_SIZE );

    // Compute the reduction array for the KF assembly
    nPartsRedKF = (int) ceil(nzTotKF/(double)nzPart);
    vector< list< list<int> > > redListPartKF( nPartsRedKF );

    // For each partition
    for( int id = 0; id < nPartsRedKF; ++id ) {
      // Take every sMem NZs so we can do a coalesced push to gmem
      int nzEnd = min(nzTotKF, (id+1)*nzPart);
      for( int nz = id*nzPart; nz < nzEnd; ++nz ) {

        if( redList[nz].size() == 0 ) continue;

        // Tack on the pointer to the NZ in the system
        // Compute a local index into smem and do a coalesced push into gmem
        int lnz = nz - id*nzPart;
        redList[nz].push_back( -lnz-1 );

        // Copy the NZ reduction list into this partition
        redListPartKF[id].push_back( redList[nz] );
      }
    }

    vector<int> redPartPtrKF;   // Partition pointer into reduction list
    vector<int> redArrayKF;     // Reduction list

    create_GPU_Arrays_LPT(redListPartKF,redPartPtrKF,redArrayKF,blockSizeRed);
    d_redPartPtrKF = redPartPtrKF;
    d_redArrayKF = redArrayKF;


    // Compute the reduction array for the F assembly
    nPartsRedF = (int) ceil(nzTotF/(double)nzPart);
    vector< list< list<int> > > redListPartF( nPartsRedF );

    // For each partition
    for( int id = 0; id < nPartsRedF; ++id ) {
      // Take every sMem NZs so we can do a coalesced push to gmem
      int nzEnd = min(nzTotF, (id+1)*nzPart);
      for( int nz = id*nzPart; nz < nzEnd; ++nz ) {

        if( redList[nz].size() == 0 ) continue;

        // NZ is already appended to reduction list

        // Copy the NZ reduction list into this partition
        redListPartF[id].push_back( redList[nz] );
      }
    }

    vector<int> redPartPtrF;   // Partition pointer into reduction list
    vector<int> redArrayF;     // Reduction list

    create_GPU_Arrays_LPT(redListPartF,redPartPtrF,redArrayF,blockSizeRed);

    d_redPartPtrF = redPartPtrF;
    d_redArrayF = redArrayF;

    /*
    //cout << nzTot << endl;
    //cout << redPartPtr << endl << endl;
    //cout << redArray << endl << endl;

    cout << "nodeArry: " << nodeArray.size() << endl;
    cout << "eIENArray: " << eIENArray.size() << endl;
    cout << "ElemData: " << (eIENPartPtr[nParts]/SIZE_EIEN)*(eDoF*(eDoF+3))/2 << endl;
    cout << "redArrayKF: " << redArrayKF.size() << endl;
    cout << "redArrayF: " << redArrayF.size() << endl;
    */

    cout << nParts << "   " << blockSize << "    " << sMemBytes << endl;
    cout << nPartsRedKF << "   " << blockSizeRed << "    " << sMemRed <<endl;
    cout << nPartsRedF << "   " << blockSizeRed << "    " << sMemRed << endl;

    d_E = vector_gpu<T>( (eIENPartPtr[nParts]/SIZE_EIEN)*(eDoF*(eDoF+3))/2 );

    d_KF = vector_gpu<T>( nzTotKF ); //vector_gpu<T>( nzTotKF );
    // Rewire: Pointer math for d_K and d_F
    d_F.setPtr( (T*) d_KF, nEq );
    d_K.setPtr( (T*) d_KF + nEq, nNZ );
  }

  // Destructor
  ~AssemblyGlobalNZ() {}

  static string name() { return "AssemblyGlobalNZ"; }


  // Perform a binary search to determine the partition number
  void findPartition()
  {
    // Get Local versions for easy
    Mesh<T>& mesh = prob.getMesh();
    const matrix<int>& IEN = mesh.getIEN();
    int nElems = mesh.nElems();
    int nNPE = mesh.nNodesPerElem();

    int maxParts = 2*4096;
    int minParts = 0;

    while( maxParts > minParts+1 ) {
      int currentParts = (maxParts + minParts)/2;

      // Partition the mesh (need to coordinate with sMem)
      cerr << "Partitioning: " << currentParts
           << "  " << minParts
           << "  " << maxParts << endl;
      mesh.partitionElems( currentParts );

      // Count the nodes required by each partition
      vector< list<int> > nPartList( currentParts );

      // For all the elements
      for( int e = 0; e < nElems; ++e ) {
        // The partition id of this element
        int id = mesh.epart[e];
        // Add this element's nodes to this partition
        for( int a = 0; a < nNPE; ++a )
          nPartList[id].push_back( IEN(e,a) );
      }

      // Find the maximum number of nodes required by a partition
      int maxN = 0;
      for( int id = 0; id < currentParts; ++id ) {
        nPartList[id].sort();
        nPartList[id].unique();
        maxN = max( (int) nPartList[id].size(), maxN );
      }

      // The GlobalNZ method is contrained by the number of nodes in sMem
      if( maxN * prob.nodalDataSize() > sMemBytes ) {
        // The partition size is too big, increase the number of parts
        minParts = currentParts;
      } else {
        // The partition size is too small, decrease the number of parts
        maxParts = currentParts;
      }
    }

    mesh.partitionElems( maxParts );
  }


  void assembleM_gpu()
  {
    M_.zero();
    dmatrix<T> m_e( prob.nEDOF() );

    // Done as a precomputation on the CPU
    int nElems = prob.mesh.nElems();

    StopWatch timer; timer.start();

    // Assemble K and F from elements
    for( int e = 0; e < nElems; ++e ) {
      //cout << e << endl;
      prob.tetrahedralMass( e, m_e );
      assemble( e, m_e, M_ );
    }
    double kernel_time = timer.stop();
    cerr << "AssemblyGlobalNZ Mass Kernel: " << kernel_time << endl;

    // Transfer to device
    d_M = (vector_cpu<T>&) M_;
  }


  void assembleKF_gpu(vector_gpu<T>& d_coord, vector_gpu<T>& d_force)
  {
    // Sanity
    //cudaMemset( d_KF, 0, (K_.size()+F_.size())*sizeof(T) );

    DEBUG_TOTAL(StopWatch_GPU timer; timer.start(););

    computeElems<__KF__,blockSize><<<nParts,blockSize,sMemBytes>>>
        ( (T*) d_E,
          (T*) d_coord, (T*) d_force,
          d_nPartPtr, d_nodeArray,
          d_eIENPartPtr, (T*) d_eIENArray );
    CHECKCUDA("GlobalNZ ElemKF");

    assembleGlobalNZ<blockSizeRed><<<nPartsRedKF,blockSizeRed,sMemRed>>>
        ( (T*) d_E, (T*) d_KF,
          d_redPartPtrKF, d_redArrayKF,
          nzPart, nzTotKF );
    CHECKCUDA("GlobalNZ AssemKF");

    INCR_TOTAL(AssemblyKF,timer.stop());
  }


  void assembleF_gpu(vector_gpu<T>& d_coord, vector_gpu<T>& d_force)
  {
    // Sanity
    //cudaMemset( d_F, 0, F_.size() * sizeof(T) );

    DEBUG_TOTAL(StopWatch_GPU timer; timer.start(););

    computeElems<__F__,blockSize><<<nParts,blockSize,sMemBytes>>>
        ( (T*) d_E,
          (T*) d_coord, (T*) d_force,
          d_nPartPtr, d_nodeArray,
          d_eIENPartPtr, (T*) d_eIENArray );
    CHECKCUDA("GlobalNZ ElemF");

    assembleGlobalNZ<blockSizeRed><<<nPartsRedF,blockSizeRed,sMemRed>>>
        ( (T*) d_E, (T*) d_KF,
          d_redPartPtrF, d_redArrayF,
          nzPart, nzTotF );
    CHECKCUDA("GlobalNZ AssemF");

    INCR_TOTAL(AssemblyF,timer.stop());
  }


};


#endif
