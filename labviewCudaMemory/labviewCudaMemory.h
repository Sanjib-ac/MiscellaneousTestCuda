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
	LABVIEWCUDAMEMORY_API void ShowLabVIEWRGB(uint64_t* LVImagePointer, int lineWidth, int height, int width);
	LABVIEWCUDAMEMORY_API void ShowImageFromInterleavedRGB_U16(const uint16_t* data, int height, int width);
	LABVIEWCUDAMEMORY_API void ShowImageFromPlanarU16(uint64_t* LVImagePointer, int lineWidth, int height, int width);
	LABVIEWCUDAMEMORY_API void ShowImageFromInterleavedU16(const uint16_t* data, int height, int width);
	LABVIEWCUDAMEMORY_API void ShowImageFromPlanarU16_LW(const uint16_t* data, int height, int width, int lineWidth);
	LABVIEWCUDAMEMORY_API void ShowImageFromInterleavedU16_LW(const uint16_t* data, int height, int width, int lineWidth);
	LABVIEWCUDAMEMORY_API void ShowImageAutoFix(const void* buffer, int dim0, int dim1, int dim2, int elemBytes);
}

#endif // LABVIEWCUDAMEMORY_H