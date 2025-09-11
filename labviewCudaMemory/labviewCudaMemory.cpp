// labviewCudaMemory.cpp : Defines the exported functions for the DLL.
//

#include "pch.h"
#include "framework.h"
#include "labviewCudaMemory.h"

#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <windows.h>
#include <cstdio>      // for FILE, freopen_s

// attached/allocated a console
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

// Macro wrapper - write CUDA_SAFE_CALL(cudaFoo(...));
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

    LABVIEWCUDAMEMORY_API void ShowImageFromPlanarRGB(unsigned char* data, int channels, int height, int width) {
        EnsureConsole();
        if (!data || channels != 3 || height <= 0 || width <= 0) return;
        const size_t planeSize = static_cast<size_t>(height) * static_cast<size_t>(width);

        // Wrap each channel plane with a cv::Mat header that doesn't copy memory.
        // Note: OpenCV uses row-major, contiguous row stride of `width` bytes here, which matches our layout.
        cv::Mat r(height, width, CV_8UC1, data + 0 * planeSize); // R plane
        cv::Mat g(height, width, CV_8UC1, data + 1 * planeSize); // G plane
        cv::Mat b(height, width, CV_8UC1, data + 2 * planeSize); // B plane

        std::vector<cv::Mat> channelsVec = { b, g, r }; // order: B, G, R
        cv::Mat img;
        cv::merge(channelsVec, img);

        cv::imshow("From LabVIEW (planar RGB)", img);
        cv::waitKey(1);
    }

    LABVIEWCUDAMEMORY_API void ShowImageFromInterleavedRGB(unsigned char* data, int height, int width) {

        if (!data || height <= 0 || width <= 0) return;
        cv::Mat rgb(height, width, CV_8UC3, data);
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        cv::imshow("From LabVIEW (interleaved RGB)", bgr);
        cv::waitKey(1);
    }

    LABVIEWCUDAMEMORY_API void ShowImageFromPlanarU16RGB(const uint16_t* data, int height, int width)
    {
        //EnsureConsole();
        if (!data || height <= 0 || width <= 0) return;

        // Compute plane size and pointers
        size_t planeSize = static_cast<size_t>(height) * width;
        const uint16_t* ptrR = data + 0 * planeSize;
        const uint16_t* ptrG = data + 1 * planeSize;
        const uint16_t* ptrB = data + 2 * planeSize;

        // Wrap U16 planes
        cv::Mat R16(height, width, CV_16UC1, const_cast<uint16_t*>(ptrR));
        cv::Mat G16(height, width, CV_16UC1, const_cast<uint16_t*>(ptrG));
        cv::Mat B16(height, width, CV_16UC1, const_cast<uint16_t*>(ptrB));

        // Sanity-check continuity and stride
        CV_Assert(R16.isContinuous() && R16.step == width * sizeof(uint16_t));
        CV_Assert(G16.isContinuous() && G16.step == width * sizeof(uint16_t));
        CV_Assert(B16.isContinuous() && B16.step == width * sizeof(uint16_t));

        // Log channel ranges
        double rMin, rMax, gMin, gMax, bMin, bMax;
        cv::minMaxLoc(R16, &rMin, &rMax);
        cv::minMaxLoc(G16, &gMin, &gMax);
        cv::minMaxLoc(B16, &bMin, &bMax);
        /*std::cout << "Ranges R:[" << rMin << "," << rMax << "] "
            << "G:[" << gMin << "," << gMax << "] "
            << "B:[" << bMin << "," << bMax << "]\n";*/

        // Prevent divide-by-zero if plane is constant
        double maxVal = std::max({ rMax, gMax, bMax, 1.0 });
        double alpha = 255.0 / maxVal;

        // Convert U16 - 8U with dynamic scaling
        cv::Mat R8, G8, B8;
        R16.convertTo(R8, CV_8UC1, alpha);
        G16.convertTo(G8, CV_8UC1, alpha);
        B16.convertTo(B8, CV_8UC1, alpha);

        // Merge to BGR and display
        std::vector<cv::Mat> bgrPlanes{ B8, G8, R8 };
        cv::Mat bgr8;
        cv::merge(bgrPlanes, bgr8);

        static const std::string winName = "From LabVIEW (planar U16 RGB)";
        cv::namedWindow(winName, cv::WINDOW_NORMAL | cv::WINDOW_GUI_EXPANDED);
        //cv::resizeWindow(winName, 800, 600);
        cv::imshow(winName, bgr8);
        cv::waitKey(1);
    }  
   
}