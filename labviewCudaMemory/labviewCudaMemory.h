#pragma once

#ifndef LABVIEWCUDAMEMORY_H
#define LABVIEWCUDAMEMORY_H

#include <cstddef>    // for size_t

#ifdef LABVIEWCUDAMEMORY_EXPORTS
#define LABVIEWCUDAMEMORY_API __declspec(dllexport)
#else
#define LABVIEWCUDAMEMORY_API __declspec(dllimport)
#endif

extern "C" {
	// Copy `count` floats from GPU memory at `devPtr` to host and print them.
	LABVIEWCUDAMEMORY_API void PrintCudaArray(const float* devPtr, size_t count);
}

#endif // LABVIEWCUDAMEMORY_H