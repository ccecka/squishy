#ifndef MESH_H
#define MESH_H

#include "General.h"
#include "MyMatrix/Matrix.h"

//#include "metis/metis.h" 

extern "C" {
typedef int idxtype;
extern void METIS_MeshToDual(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
extern void METIS_MeshToNodal(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
extern void METIS_PartGraphKway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
}



/* Isoparametric Mesh Class */

template <typename T>
class Mesh
{
  matrix<T> coord;   // Nodal Coordinates:  GNode #, Dim to Coord
  matrix<int> IEN;   // Connectivity Matrix: Elem #, LNode # to GNode #

 public:

  vector<int> nxadj;       // Nodal adjacency pointer
  vector<int> nadjncy;     // Nodal adjacency array
 
  vector<int> dxadj;       // Dual adjacency pointer
  vector<int> dadjncy;     // Dual adjacency array
 
  vector<int> npart;       // Nodal partition index array
  vector<int> epart;       // Elemental partition index array
 
  int etype;               // The type of element
  int nVN;                 // The total number of vertex nodes in the mesh
 
  // Constructors
 Mesh( matrix<T>& coord_, matrix<int>& IEN_ )
   : coord( coord_ ), IEN( IEN_ ),
    nxadj( nNodes()+1 ), nadjncy( 15*nNodes() ),
    dxadj( nElems()+1 ), dadjncy( 4*nElems() ),
    npart( nNodes(), -1 ), epart( nElems(), -1 )
    {
      if( nNodesPerElem() == 3 ) {
	etype = 1;           // METIS tri
      } else if( nNodesPerElem() == 4 ) {
	etype = 2;           // METIS tet
      } else {
	cerr << "Element Type Init Fail" << endl;
	exit(0);
      }

      /*
      // Remove hanging nodes (Damn you Raymond)
      vector<int> node_count( nNodes(), 0 );
      for( int k = 0; k < IEN.size(); ++k ) {
	++node_count[ IEN[k] ];
      }
      int cumsum = 0;
      for( int k = 0; k < nNodes(); ++k ) {
	if( node_count[k] == 0 ) {
	  ++cumsum;
	}
	node_count[k] = cumsum;
      }
      // Shift the coords
      int numNodes = nNodes() - node_count[nNodes()-1];
      matrix<T> newCoord(numNodes,nDim());
      for( int n = 0; n < numNodes; ++n ) {
	newCoord(n,0) = coord(n + node_count[n],0);
	newCoord(n,1) = coord(n + node_count[n],1);
	newCoord(n,2) = coord(n + node_count[n],2);
      }
      coord = newCoord;
      // Shift the IEN node numbers
      for( int k = 0; k < IEN.size(); ++k ) {
	IEN[k] = IEN[k] - node_count[ IEN[k] ];
      }
      */
      
      int nE = nElems();
      int nN = nNodes();
      nVN = nN;           // Can be used for higher orders (First order now)

      int pnumflag = 0;   // C-style numbering
      

      METIS_MeshToNodal(&nE, &nN, const_cast<idxtype*>(&IEN[0]), &etype,
			&pnumflag, &nxadj[0], &nadjncy[0]);
      
      METIS_MeshToDual(&nE, &nN, const_cast<int*>(&IEN[0]), &etype,
		       &pnumflag, &dxadj[0], &dadjncy[0]);


      // Make sure there are no hanging nodes... and find smallest edge
      double smallE = 10000;
      for( int n = 0; n < nNodes(); ++n ) {
	if( nxadj[n+1] - nxadj[n] <= 0 ) {
	  cout << n << endl;
	  //assert( nxadj[n+1] - nxadj[n] > 0 );
	}
	// For all neighbor nodes
	for( int nnPtr = nxadj[n]; nnPtr < nxadj[n+1]; ++nnPtr ) {
	  int nn = nadjncy[nnPtr];
	  if( nn < n ) continue;
	  double dx = coord(n,0) - coord(nn,0);
	  double dy = coord(n,1) - coord(nn,1);
	  double dz = coord(n,2) - coord(nn,2);
	  smallE = min(smallE, sqrt(dx*dx + dy*dy + dz*dz));
	}
      }
      cerr << "Smallest Edge: " << smallE << endl;
    }
  
  inline matrix<T>& getCoord()
  {
    return coord;
  }

  inline matrix<int>& getIEN()
  {
    return IEN;
  }

  inline int nVertexNodes()
  {
    return nVN;
  }

  inline int nNodes()
  {
    return coord.nRows();
  }

  inline int nDim()
  {
    return coord.nCols();
  }

  inline int nElems()
  {
    return IEN.nRows();
  }

  inline int nNodesPerElem()
  {
    return IEN.nCols();
  }

  inline int nVertexNodesPerElem()
  {
    if( etype == 1 )
      return 3;           // Tri
    else if( etype == 2 )
      return 4;           // Tet
    else {
      cerr << "etype Error" << endl;
      exit(0);
    }
  }

  inline int nFacesPerElem()
  {
    if( etype == 1 )
      return 3;           // Tri
    else if( etype == 2 )
      return 4;           // Tet
    else {
      cerr << "etype Error" << endl;
      exit(0);
    }
  }

  inline int nNodesPerFace()
  {
    if( etype == 1 )
      return 2;           // Tri
    else if( etype == 2 )
      return 3;           // Tet
    else {
      cerr << "etype Error" << endl;
      exit(0);
    }
  }

  inline vector<int>& partitionElems( int N )
  {
    if( N == 1 ) {   // METIS doesn't seem to like doing a null op
      epart.assign( epart.size(), 0 );
    } else {
      int edgecut = 0;              // return value
      
      int nE = nElems();
      
      int pnumflag = 0;             // C-style numbering
      vector<int> options(5,0);     // Default all options
      int wgtflag = 0;              // No weights
      
      METIS_PartGraphKway(&nE, &dxadj[0], &dadjncy[0], NULL, NULL, 
			  &wgtflag, &pnumflag, &N, &options[0], 
			  &edgecut, &epart[0]);
    }
    
    return epart;
  }
  
  inline vector<int>& partitionNodes( int N )
  {
    if( N == 1 ) {   // METIS doesn't seem to like doing a null op
      npart.assign( npart.size(), 0 );
    } else if( N >= nVertexNodes() ) {
      for( int n = 0; n < nVertexNodes(); ++n ) {
	npart[n] = n;
      }
    } else {
      int edgecut = 0;              // return value
      
      int vN = nVertexNodes();
      
      int pnumflag = 0;             // C-style numbering
      vector<int> options(5,0);     // Default all options
      int wgtflag = 0;              // No weights
      
      METIS_PartGraphKway(&vN, &nxadj[0], &nadjncy[0], NULL, NULL, 
			  &wgtflag, &pnumflag, &N, &options[0], 
			  &edgecut, &npart[0]);
    }
    
    return npart;
  }
  
};


#endif
