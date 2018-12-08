#ifndef _errorMacros_hu_
#define _errorMacros_hu_

#include <stdio.h>
#include "Error.h"

#define SPIT(...) \
{ \
	fprintf(stderr, ##__VA_ARGS__); \
	throw Error(); \
}

#define CHECK(X) \
	if(X) \
		SPIT("Error: condition %s not met at %s:%d\n", #X, __FILE__, __LINE__); \

#define ASSERT(X) CHECK(!X)

#define CHECK_SUCCESS(X) \
	if(X != cudaSuccess) \
		SPIT("Error: %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__); \

#endif // _errorMacros_hu_
