#ifndef PARSER_H
#define PARSER_H

#include <fstream>
#include <sys/stat.h>

#include "MyMatrix/Vector.h"
#include "MyMatrix/Matrix.h"

// Returns the file size in the number of bytes
int file_size( const char* filename )
{
  std::ifstream f;
  f.open( filename, std::ios_base::binary | std::ios_base::in );
  if( !f.good() || f.eof() || !f.is_open() ) 
    return 0;
  f.seekg(0, std::ios_base::beg);
  std::ifstream::pos_type begin_pos = f.tellg();
  f.seekg(0, std::ios_base::end);
  std::ifstream::pos_type end_pos = f.tellg();
  f.close();
  return static_cast<int>(end_pos - begin_pos);
}

// Returns a bool indicating whether fileName exists
bool file_exists( const char* fileName )
{
  struct stat buf;
  int i = stat( fileName, &buf );
  return (i == 0);
}


/** Writers **/

template <class T>
void writeBIN(T* a, int N, const char* filename)
{
  fstream myFile(filename, ios::out | ios::binary);
  myFile.write( (char*) a, N*sizeof(T) );
  myFile.close();
}

template <class T>
void writeBIN(vector<T>& a, const char* filename) 
{
  writeBIN<T>( &a[0], a.size(), filename );
}


/** Readers **/

template <class T>
void parseBIN(const char* filename, T* a, int N)
{
  fstream myFile(filename, ios::in | ios::binary);
  myFile.read( (char*) a, N*sizeof(T) );
  myFile.close();
}

template <class T>
void parseBIN(const char* filename, vector<T>& a)
{
  int N = file_size( filename ) / sizeof(T);
  a.resize(N);
  parseBIN<T>(filename, &a[0], N);
}

template <class T>
void parseBIN( const char* filename, matrix<T>& M, int nCols )
{
  int N = file_size(filename) / sizeof(T);
  M = matrix<T>( (int) ceil(N/(double)nCols), nCols );
  parseBIN<T>(filename, M.val_array(), N);
}

 
template <class T>
void parseOFF( const char* filename, 
	       matrix<T>& coord, int nDim,
	       matrix<int>& IEN, int nNPE )
{
  ifstream inFile;
  inFile.open( filename );

  // Read the OFF header
  string temp;
  inFile >> temp;

  // Read the data header
  int nNodes;
  inFile >> nNodes;
  int nElems;
  inFile >> nElems;
  int N;
  inFile >> N;

  // Read the coord array
  double x;
  coord = matrix<T>( nNodes, nDim );
  for( int n = 0; n < nNodes; ++n ) {
    for( int d = 0; d < nDim; ++d ) {
      inFile >> x;
      coord(n,d) = (T) x;
    }
  }

  // Read the elem array
  int nNPE_;
  IEN = matrix<int>( nElems, nNPE );
  for( int e = 0; e < nElems; ++e ) {
    inFile >> nNPE_;
    assert( nNPE_ == nNPE );
    for( int n = 0; n < nNPE; ++n ) {
      inFile >> N;
      IEN(e,n) = (int) N;
    }
  }
}

template <class T>
void parseDAT( const char* filename, 
	       matrix<T>& coord, int nDim,
	       matrix<int>& IEN, int nNPE )
{
  cout << filename << endl;

  ifstream inFile;
  inFile.open( filename );

  // Read the OFF header
  string temp;
  inFile >> temp;

  // Read the data header
  int nNodes;
  inFile >> nNodes;
  int nElems;
  inFile >> nElems;
  int N;
  inFile >> N;

  // Read the coord array
  double x;
  coord = matrix<T>( nNodes, nDim );
  for( int n = 0; n < nNodes; ++n ) {
    for( int d = 0; d < nDim; ++d ) {
      inFile >> x;
      coord(n,d) = (T) x;
    }
  }

  // Read the elem array
  IEN = matrix<int>( nElems, nNPE );
  for( int e = 0; e < nElems; ++e ) {
    for( int n = 0; n < nNPE; ++n ) {
      inFile >> N;
      IEN(e,n) = N - 1;
    }
  }
}

#endif
