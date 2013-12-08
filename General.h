#ifndef GENERAL_H
#define GENERAL_H

// IO
#include <iostream>
#include <iomanip>
#include <sstream>

// STL
#include <cmath>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <queue>
#include <deque>


// Define the precision (float or double)   (double sorta-untested!)
#define MY_REAL float

// Lame1     lambda
const static MY_REAL BULK_MOD = 2;

// Lame2     mu     Shear Modulus
const static MY_REAL SHEAR_MOD = 5;

// Mass      rho
const static MY_REAL MASS = 1;

// Time Step
const static MY_REAL DT = 0.009;

// Damping [0,1]
const static MY_REAL DAMPING = 0.01;

// Orientation   +/- 1   to change normal
const static int ORIENT = -1;

// The CUDA warp size
const static int WARP_SIZE = 32;


// DEBUG
#include <typeinfo>
// Define to remove assert and debug statements
//#define NDEBUG
#include <assert.h>
#ifdef NDEBUG
  /* helper for code location */
  #define LOC std::cerr << __FILE__ << ":" << __LINE__ << "  "
  /* macro for general debug print statements. */
  #define COUT_TXT(text) LOC; std::cerr << text << std::endl
  /* macro that dumps a variable name and its value */
  #define COUT_VAR(var) LOC; std::cerr << (#var) << " = " << var << std::endl
#else
  #define LOC
  #define COUT_TXT(text)
  #define COUT_VAR(var)
#endif


// Define to provide timing each section of code
//#define TOTAL_TIMING

#ifdef TOTAL_TIMING

#define DEBUG_TOTAL(s) s
#define INCR_TOTAL(STAGE,TIME) total_##STAGE##_Time += TIME; \
                               ++total_##STAGE##_Iter
#define COUT_TOTAL(STAGE) cerr << "Total " << #STAGE << " Time: " \
                               << total_##STAGE##_Time << "/" \
                               << total_##STAGE##_Iter << " = " \
                               << total_##STAGE##_Time/total_##STAGE##_Iter \
                               << endl

double total_NR_Time = 0;
int total_NR_Iter = 0;
double total_AssemblyKF_Time = 0;
int total_AssemblyKF_Iter = 0;
double total_AssemblyF_Time = 0;
int total_AssemblyF_Iter = 0;
double total_CG_Time = 0;
int total_CG_Iter = 0;
double total_MVM_Time = 0;
int total_MVM_Iter = 0;
double total_Transfer_Time = 0;
int total_Transfer_Iter = 0;
double total_Frame_Time = 0;
int total_Frame_Iter = 0;

#else

#define DEBUG_TOTAL(s)
#define INCR_TOTAL(STAGE,TIME)
#define COUT_TOTAL(STAGE)

#endif






// Macro magic
#define VAR_MANIPULATOR_(PRE,VAR,POST) PRE##VAR##POST
#define VAR_MANIPULATOR(PRE,VAR,POST) VAR_MANIPULATOR_(PRE,VAR,POST)

// Macro magic to make the corresponding GL_REAL type
#define GL_float  GL_FLOAT
#define GL_double GL_DOUBLE
#define GL_REAL VAR_MANIPULATOR(GL_,MY_REAL,)

// Macro magic to make the corresponding REAL3 type for CUDA
#define REAL3 VAR_MANIPULATOR(,REAL,3)


using namespace std;


inline bool ISODD( int a )
{
  return (a & 1);
}

inline bool ISEVEN( int a )
{
  return !ISODD(a);
}

// ceil(a/b) for integers
inline int DIVIDE_INTO( int a, int b )
{
  return (a+b-1)/b;
}

// Round up to the next multiple of b
template <typename T>
inline int round_up( T a, int b )
{
  return b*(int)ceil(a/float(b));
}

// Round down to the next multiple of b
template <typename T>
inline int round_down( T a, int b )
{
  return b*(int)floor(a/float(b));
}


// Convert primitives to strings
template <typename T>
inline std::string toString( T a )
{
  std::stringstream ss;
  ss << a;
  return ss.str();
}

// Comparison operator for objects with .size() function
template <class A>
bool compare_size( A& a, A& b )
{
  return (a.size() < b.size());
}


#include <sys/time.h>

class StopWatch
{
  timeval startTime, stopTime, diffTime;
 public:
  StopWatch() { start(); }
  inline void start() { gettimeofday(&startTime,NULL); }
  inline double stop() { return elapsed(); }
  inline double elapsed() {
    gettimeofday(&stopTime,NULL);
    timersub(&stopTime, &startTime, &diffTime);
    return diffTime.tv_sec + diffTime.tv_usec/1000000.0; // 10^6 uSec per Sec
  }
};

typedef StopWatch StopWatch_CPU;



#endif
