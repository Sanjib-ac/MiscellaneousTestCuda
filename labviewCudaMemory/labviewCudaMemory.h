#pragma once

#ifndef LABVIEWCUDAMEMORY_H
#define LABVIEWCUDAMEMORY_H

#include <cstddef>    // for size_t
#include <cstdint>    // for uint16_t

#ifdef LABVIEWCUDAMEMORY_EXPORTS
#define LABVIEWCUDAMEMORY_API __declspec(dllexport)
#else
#define LABVIEWCUDAMEMORY_API __declspec(dllimport)
#endif

extern "C" {
	// Copy `count` floats from GPU memory at `devPtr` to host and print them.
	LABVIEWCUDAMEMORY_API void PrintCudaArray(const float* devPtr, size_t count);
	LABVIEWCUDAMEMORY_API void ShowImageFromPlanarRGB(unsigned char* data, int channels, int height, int width);
	LABVIEWCUDAMEMORY_API void ShowImageFromInterleavedRGB(unsigned char* data, int height, int width);
	LABVIEWCUDAMEMORY_API void ShowImageFromPlanarU16RGB(const uint16_t* data, int height, int width);
}

#endif // LABVIEWCUDAMEMORY_H