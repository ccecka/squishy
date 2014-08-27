#ifndef ASSEMBLYSHAREDNZ_H
#define ASSEMBLYSHAREDNZ_H

#include "Assembly_Interface.h"

#include "../Parser.h"

#include "AssemblyUtil.h"
#include "AssemblySharedNZ.cu"

#include "elemRepo/kernelMapF_e.ker"
#include "elemRepo/kernelMapK_e.ker"


template <typename T>
class AssemblySharedNZ : public AssemblyGPU<T>
{
  using Assembly_Interface<T>::prob;
  using Assembly_Interface<T>::ID;
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
  const static int blockSize = 160;   // CUDA blockSize
  int sMemBytes;              // CUDA shared Memory

  vector<int> scatterPartPtr;    // Partition pointer into scatter array
  vector<int> scatterArray;      // Partition scatter array

  vector_gpu<int> d_scatterPartPtr;         // Device pointer to nPartPtr
  vector_gpu<int> d_scatterArray;           // Device pointer to nPart

  vector<int> eNumPart;        // The number of elements for each partition

  vector_gpu<int> d_eNumPart;

  vector<int> suppPtr;          // Data needed by each element
  vector<T> suppArray;

  vector_gpu<int> d_suppPtr;
  vector_gpu<T> d_suppArray;

  // KF assembly
  int nzTotKF;                // The total number of nz to assemble

  vector<int> redPartPtrKF;   // Partition pointer into reduction list
  vector<int> redArrayKF;     // Reduction list
  vector_gpu<int> d_redPartPtrKF;        // Device pointer to redPartPtr
  vector_gpu<int> d_redArrayKF;          // Device pointer to redList

  // F assembly
  int nzTotF;                 // The total number of nz to assemble

  vector<int> redPartPtrF;   // Partition pointer into reduction list
  vector<int> redArrayF;     // Reduction list
  vector_gpu<int> d_redPartPtrF;        // Device pointer to redPartPtr
  vector_gpu<int> d_redArrayF;          // Device pointer to redList

 public:

  double cuda_time1;
  double cuda_time2;


  // Everything that we can do as a precomputation
  template <template <typename> class MATRIX>
  AssemblySharedNZ( Problem<T>& p, MATRIX<T>& FEM_Matrix )
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
    int vNPE = mesh.nVertexNodesPerElem();
    int nNodes = mesh.nNodes();

    // Fudge factor of 100
    sMemBytes = cudaMaxSMEM() - 100;

    // Load or compute the nodal partition mesh.npart
    string fileStr = "data/SharedNZ_" + toString(nNodes)
	+ "_" + toString(sizeof(T)) + ".part";
    const char* filename = fileStr.c_str();
    if( file_exists(filename) ) {
      cout << "Reading Partition From " << fileStr << endl;
      parseBIN( filename, mesh.npart );
    } else {
      findPartition();
      writeBIN( mesh.npart, filename );
    }

    nParts = max( mesh.npart ) + 1;

    // Get the elements and nodes needed by each partition
    vector< list<int> > ePartList( nParts );
    vector< list<int> > nPartList( nParts );
    vector< list<int> > nzPartList( nParts );
    // For all the elements
    for( int e = 0; e < nElems; ++e ) {
      // For each vertex node of the element
      for( int a = 0; a < vNPE; ++a ) {
        // Get the partition's array
        int n = IEN(e,a);
        list<int>& eList = ePartList[ mesh.npart[n] ];
        list<int>& nList = nPartList[ mesh.npart[n] ];
        list<int>& nzList = nzPartList[ mesh.npart[n] ];

        // This element is needed by this partition
        eList.push_back(e);
        // All the element's nodes are needed by this partition
        for( int a2 = 0; a2 < vNPE; ++a2 )
          nList.push_back( IEN(e,a2) );

        // This partition is responsible for all of the NZs of this node
        // For each dof of this node
        for( int d = 0; d < nDoF; ++d ) {
          // Include all the NZs in F
          int eq1 = ID(n,d);
          nzList.push_back( nNZ + eq1 );
          // Include all the NZs contributed by this element
          for( int k2 = 0; k2 < eDoF; ++k2 ) {
            int eq2 = LM(e,k2);
            nzList.push_back( K_.IJtoK( eq1, eq2 ) );
          }
        }
      }
    }

    // Uniquify the nodes, elements, and NZs needed by each partition
    for( int id = 0; id < nParts; ++id ) {
      ePartList[id].sort();
      ePartList[id].unique();
      nPartList[id].sort();
      nPartList[id].unique();
      nzPartList[id].sort();
      nzPartList[id].unique();
    }

    // Get statistics on this partitioning
    int totE = 0;
    int maxE = 0;
    int minE = 1000000;
    eNumPart = vector<int>( nParts );
    int totN = 0;
    int maxN = 0;
    int minN = 1000000;
    // For each partition
    for( int id = 0; id < nParts; ++id ) {
      int nE = ePartList[id].size();
      totE += nE;
      maxE = max( maxE, nE );
      minE = min( minE, nE );
      eNumPart[id] = nE;

      int nN = nPartList[id].size();
      totN += nN;
      maxN = max( maxN, nN );
      minN = min( minN, nN );
    }

    cout<<"nParts: "<<nParts<<endl;
    cout<<"MinE: "<<minE<<"\tMaxE: "<<maxE<<"\tTotE: "<<totE<<endl;
    cout<<"MinN: "<<minN<<"\tMaxN: "<<maxN<<"\tTotN: "<<totN<<endl;

    // Make sure this will fit in sMem
    assert( maxE * prob.elemDataSize() < sMemBytes );

    // Create elemental and nodal partition maps
    vector< map<int,int> > n2lMapPart( nParts );
    vector< map<int,int> > e2lMapPart( nParts );
    for( int id = 0; id < nParts; ++id ) {
      list<int>& eList = ePartList[id];
      // Create a global element to partition element map
      map<int,int>& e2lMap = e2lMapPart[id];
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

    // CREATE THE SCATTER LIST
    vector< list< list<int> > > scatterList( nParts );

    // For each partition
    for( int id = 0; id < nParts; ++id ) {
      list<int>& nList = nPartList[id];
      list<int>& eList = ePartList[id];

      // For each node needed by this partition
      list<int>::iterator ni;
      for( ni = nList.begin(); ni != nList.end(); ++ni ) {
        int n = *ni;

        // Start a scatter array with this node's index
        list<int> nScatter;
        nScatter.push_back( -n-1 );

        // For each element of this partition
        list<int>::iterator ei;
        for( ei = eList.begin(); ei != eList.end(); ++ei ) {
          int e = *ei;

          // For each vertex node of this element
          for( int a = 0; a < vNPE; ++a ) {

            // If this element node is the node in question
            if( IEN(e,a) == n ) {
              // Compute a target in sMem to scatter this data to
              int el = e2lMapPart[id][e];
              int target = el * prob.elemDataSize() / sizeof(T) +
                  a * prob.nodalDataSize() / sizeof(T);

              nScatter.push_back( target + 1 );
            }

          }

        }

        // Done creating this node's scatter array
        scatterList[id].push_back( nScatter );

      }
    }

    // Create coalseced GPU Arrays for reading nodal data to smem
    create_GPU_Arrays_LPT( scatterList, scatterPartPtr, scatterArray,
                           blockSize );


    // NEED TO LAY OUT THE JINV DATA FOR FAST READING
    vector< list< vector<T> > > suppData( nParts );
    // For each partition
    for( int id = 0; id < nParts; ++id ) {
      // For each element of this partition
      list<int>& eList = ePartList[id];
      list<int>::iterator ei;
      for( ei = eList.begin(); ei != eList.end(); ++ei ) {
        int e = *ei;

        // Create a vector of the suppData for this element
        suppData[id].push_back( prob.getSuppData(e) );
      }
    }

    // Create coalesced GPU Array for reading supplemental data
    create_GPU_Arrays( suppData, suppPtr, suppArray, blockSize );


    // Construct the reduction array for each NZ of the system
    nzTotKF = nNZ + nEq;
    nzTotF = nEq;
    vector< list<int> > redList( nzTotKF );

    // For all the elements
    for( int e = 0; e < nElems; ++e ) {

      // For each node of this element
      for( int a = 0; a < vNPE; ++a ) {
        // Get the global node number
        int n = IEN(e,a);
        // Get the partition number responsible for this node
        int id = mesh.npart[n];
        // Get the e2l map for this partition
        map<int,int>& e2lMap = e2lMapPart[id];
        // Get the local element number of e in partition id
        int le = e2lMap[e];

        // Get the location in shared memory that this element data begins
        // Symmetric Element Data position in shared memory
        int sMemStart = le * ((eDoF*(eDoF+3))/2);

        // For each dof of this node
        for( int d = 0; d < nDoF; ++d ) {
          int k1 = a*nDoF + d;
          int eq1 = LM(e,k1);
          if( eq1 == -1 ) continue;

          // For each column of the element data
          for( int k2 = 0; k2 < eDoF; ++k2 ) {
            int eq2 = LM(e,k2);
            if( eq2 == -1 ) continue;

            // If symmetric and only assembling half the matrix
            //if( eq2 <= eq1 ) continue;

            // Get the NZ of K that this element will be accumulated into
            int nz = K_.IJtoK( eq1, eq2 );

            // Get the location in global memory of this element data entry
            int data = sMemStart + kernelMapK_e( k1, k2 );

            redList[nz].push_back( data+1 );
          }

          // The F[eq] of the forcing vector for this element
          int nz = nNZ + eq1;

          // Get the location in global memory of this data entry
          int data = sMemStart + kernelMapF_e( k1 );

          redList[nz].push_back( data+1 );
        }
      }
    }

    // Append the -nz-1 to the end of the reduction list
    for( int nz = 0; nz < nzTotKF; ++nz ) {
      redList[nz].sort();
      redList[nz].push_back( -nz-1 );
    }

    // Compute the reduction array for the KF assembly
    vector< list< list<int> > > redListPartKF( nParts );

    // For each partition
    for( int id = 0; id < nParts; ++id ) {
      list<int>& nzList = nzPartList[id];
      // For all the NZs this partition is responsible for
      list<int>::iterator li;
      for( li = nzList.begin(); li != nzList.end(); ++li ) {
        redListPartKF[id].push_back( redList[*li] );
      }
    }

    create_GPU_Arrays_LPT(redListPartKF,redPartPtrKF,redArrayKF,blockSize);

    // Compute the reduction array for the F assembly
    vector< list< list<int> > > redListPartF( nParts );

    // For each partition
    for( int id = 0; id < nParts; ++id ) {
      list<int>& nzList = nzPartList[id];
      // For all the NZs this partition is responsible for
      list<int>::iterator li;
      for( li = nzList.begin(); li != nzList.end(); ++li ) {
        if( *li >= nNZ )
          redListPartF[id].push_back( redList[*li] );
      }
    }

    create_GPU_Arrays_LPT(redListPartF,redPartPtrF,redArrayF,blockSize);


    cout << nParts << "   " << blockSize << "    " << sMemBytes << endl;

    d_scatterPartPtr = scatterPartPtr;
    d_scatterArray = scatterArray;
    d_eNumPart = eNumPart;
    d_suppPtr = suppPtr;
    d_suppArray = suppArray;

    d_KF = vector_gpu<T>( nzTotKF ); //vector_gpu<T>( nzTotKF );
    // Rewire: Pointer math for d_K and d_F
    d_K.setPtr( (T*) d_KF, nNZ );
    d_F.setPtr( (T*) d_KF + nNZ, nEq );

    d_redPartPtrKF = redPartPtrKF;
    d_redArrayKF = redArrayKF;
    d_redPartPtrF = redPartPtrF;
    d_redArrayF = redArrayF;
  }

  // Destructor
  ~AssemblySharedNZ() {}

  static string name() { return "AssemblySharedNZ"; }

  // Determine a partition
  void findPartition()
  {
    // Get Local versions for easy
    Mesh<T>& mesh = prob.getMesh();
    const matrix<int>& IEN = mesh.getIEN();
    int nElems = mesh.nElems();
    //int nNPE = mesh.nNodesPerElem();
    int vNPE = mesh.nVertexNodesPerElem();
    int nNodes = mesh.nNodes();

    // Partition the nodes
    //mesh.partitionNodes( nParts );

    // METIS doesn't like makng very small partitions
    // Here, we run a quick greedy algorithm to group neighbor nodes
    // which have few adjacent elements.

    // Assign each node its own partition
    for( int n = 0; n < nNodes; ++n ) {
      mesh.npart[n] = n;
    }
    nParts = nNodes;

    // Get the elements needed by each partition
    vector< list<int> > ePartList( nParts );
    // For all the elements
    for( int e = 0; e < nElems; ++e ) {
      // For each vertex node of the element
      for( int a = 0; a < vNPE; ++a ) {
	// Get the partition's array
	int n = IEN(e,a);
	list<int>& eList = ePartList[ mesh.npart[n] ];

	// If we haven't already added this element to this partition
	if( eList.back() == e ) continue;

	// This element is needed by this partition
	eList.push_back(e);
      }
    }

    // Greedy algorithm to reduce the number of partitions

    int lastnParts = -1;
    while( nParts != lastnParts ) {
      lastnParts = nParts;

      // For each node
      for( int n = 0; n < nNodes; ++n ) {
	int idn = mesh.npart[n];

	// Find the neighbor node, not in this partition, with the
	// smallest number of adjacent elements
	int smallestCombinedE = 100000;
	int smallestNN = -1;

	for( int nnPtr = mesh.nxadj[n]; nnPtr < mesh.nxadj[n+1]; ++nnPtr ) {
	  int nn = mesh.nadjncy[nnPtr];
	  int idnn = mesh.npart[nn];
	  if( idnn == idn ) continue;

	  // Try adding this neighbor node to this partition,
	  // staying under the sMem requirement
	  set<int> combinedE;
	  combinedE.insert( ePartList[idnn].begin(),
			    ePartList[idnn].end() );
	  combinedE.insert( ePartList[idn].begin(),
			    ePartList[idn].end() );

	  if( int(combinedE.size()) * prob.elemDataSize() < sMemBytes ) {
	    // This fits, if it's smaller than the rest, keep it
	    if( int(combinedE.size()) < smallestCombinedE ) {
	      smallestCombinedE = combinedE.size();
	      smallestNN = nn;
	    }
	  }

	}

	// No neighbor node worked
	if( smallestNN == -1 ) continue;

	// Add this neighbor node to this partition!
	int nn = smallestNN;
	int idnn = mesh.npart[nn];

	//cerr << "Assigning Node " << nn
	//     << " to Partition " << idn
	//     << " with size " << smallestCombinedE << endl;

	// Combine into the original partition number
	ePartList[idn].insert( ePartList[idn].end(),
			       ePartList[idnn].begin(),
			       ePartList[idnn].end() );
	ePartList[idn].sort();
	ePartList[idn].unique();
	// Assign the partition number idnn to idn
	for( int ns = 0; ns < nNodes; ++ns ) {
	  if( mesh.npart[ns] == idnn )
	    mesh.npart[ns] = idn;
	}
	// One less partition
	--nParts;
	//cout << "nParts: " << nParts << endl;
      }

    }

    // Renumber the partitions
    vector<int> id_count(nNodes,0);
    for( int n = 0; n < nNodes; ++n )
      ++id_count[ mesh.npart[n] ];
    // Accumulate offsets for the number of zeros that occur
    int zeros = 0;
    for( int id = 0; id < nNodes; ++id ) {
      if( id_count[id] == 0 ) ++zeros;
      id_count[id] = zeros;
    }
    // Apply offsets and adjust N
    for( int n = 0; n < nNodes; ++n ) {
      mesh.npart[n] -= id_count[ mesh.npart[n] ];
    }
  }


  void assembleM_gpu()
  {
    M_.zero();
    dmatrix<T> m_e( prob.nEDOF() );

    // Done as a precomputation on the CPU
    int nElems = prob.getMesh().nElems();

    //StopWatch timer; timer.start();

    // Assemble K and F from elements
    for( int e = 0; e < nElems; ++e ) {
      //cout << e << endl;
      prob.tetrahedralMass( e, m_e );
      this->assemble( e, m_e, M_ );
    }

    //double kernel_time = timer.stop();
    //cerr << "AssemblySharedNZ Mass Kernel: " << kernel_time << endl;

    // Transfer to device
    d_M = (vector_cpu<T>&) M_;
  }


  void assembleKF_gpu( vector_gpu<T>& d_coord, vector_gpu<T>& d_force )
  {
    // Sanity Check
    //cudaMemset( d_KF, 0, (K_.size()+F_.size()) * sizeof(T) );

    DEBUG_TOTAL(StopWatch_GPU timer; timer.start(););

    assembleSharedNZ<__KF__,blockSize><<<nParts,blockSize,sMemBytes>>>
        ( (T*) d_coord, (T*) d_force,
          d_scatterPartPtr, d_scatterArray,
          d_eNumPart,
          d_suppPtr, (T*) d_suppArray,
          d_redPartPtrKF, d_redArrayKF,
          (T*) d_KF );

    CHECKCUDA("SharedNZ AssemKF");

    //cuda_time1 = timer.stop();
    //cerr << "AssemblySharedNZ KernelKF: " << cuda_time1 << endl;

    INCR_TOTAL(AssemblyKF,timer.stop());
  }


  void assembleF_gpu( vector_gpu<T>& d_coord, vector_gpu<T>& d_force )
  {
    // Sanity check
    //cudaMemset( d_F, 0, F_.size() * sizeof(T) );

    DEBUG_TOTAL(StopWatch_GPU timer; timer.start(););

    assembleSharedNZ<__F__,blockSize><<<nParts,blockSize,sMemBytes>>>
        (	(T*) d_coord, (T*) d_force,
                d_scatterPartPtr, d_scatterArray,
                d_eNumPart,
                d_suppPtr, (T*) d_suppArray,
                d_redPartPtrF, d_redArrayF,
                (T*) d_KF );

    CHECKCUDA("SharedNZ AssemF");

    //cuda_time1 = timer.stop();
    //cerr << "AssemblySharedNZ KernelF: " << cuda_time1 << endl;

    INCR_TOTAL(AssemblyF,timer.stop());
  }


};


#endif
