// labviewCudaMemory.cpp : Defines the exported functions for the DLL.
//

#include "pch.h"
#include "framework.h"
#include "labviewCudaMemory.h"

#include <cuda_runtime.h>
#include <iostream>
#include <windows.h>
#include <cstdio>      // for FILE, freopen_s

// Tracks whether we've already attached/allocated a console
static bool console_initialized = false;

// Ensures a console is available for stdout/stderr
static void EnsureConsole() {
    if (!console_initialized) {
        // Try attaching to parent; if that fails, create a new one
        if (!AttachConsole(ATTACH_PARENT_PROCESS)) {
            AllocConsole();
        }
        FILE* stream = nullptr;
        freopen_s(&stream, "CONOUT$", "w", stdout);
        freopen_s(&stream, "CONOUT$", "w", stderr);
        freopen_s(&stream, "CONIN$", "r", stdin);

        std::cout << "Console initialized!" << std::endl;
        console_initialized = true;
    }
}

// Inline helper to check CUDA errors (captures file+line)
inline void cudaSafeCall(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "[PrintCudaArray] CUDA error at "
            << file << ":" << line << " - "
            << cudaGetErrorString(err) << "\n";
        return;
    }
}

// Macro wrapper so you can write CUDA_SAFE_CALL(cudaFoo(...));
#define CUDA_SAFE_CALL(call) cudaSafeCall((call), __FILE__, __LINE__)

extern "C" {

    LABVIEWCUDAMEMORY_API void PrintCudaArray(const float* devPtr, size_t count) {
        EnsureConsole();
        if (!devPtr || count == 0) {
            std::cerr << "[PrintCudaArray] Invalid pointer or zero count\n";
            return;
        }        

        // Allocate pinned host memory
        float* hostBuf = nullptr;
        CUDA_SAFE_CALL(cudaMallocHost(
            reinterpret_cast<void**>(&hostBuf),
            count * sizeof(float)
        ));

        // Copy device host
        CUDA_SAFE_CALL(cudaMemcpy(
            hostBuf,
            devPtr,
            count * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        // Print each element
        for (size_t i = 0; i < count; ++i) {
            std::cout << "Element[" << i << "] = " << hostBuf[i] << "\n";
        }

        cudaFreeHost(hostBuf);
    }

}