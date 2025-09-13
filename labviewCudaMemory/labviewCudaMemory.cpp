// labviewCudaMemory.cpp : Defines the exported functions for the DLL.
//

#include "pch.h"
#include "framework.h"
#include "labviewCudaMemory.h"

#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

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

// Default constants
static constexpr int   DEF_WIDTH = 1280;
static constexpr int   DEF_HEIGHT = 2880;
static constexpr float DEF_R = 0.10f;
static constexpr float DEF_G = 0.11f;
static constexpr float DEF_B = 0.13f;

// Clamp [lo, hi]
static inline int clamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Convert float → 8-bit with rounding & saturation
static inline unsigned char toU8(float v) {
    int iv = static_cast<int>(v + 0.5f);
    iv = std::max(0, std::min(255, iv));
    return static_cast<unsigned char>(iv);
}

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

    LABVIEWCUDAMEMORY_API void ShowImageFromPlanarU16RGB_CUDA(const uint16_t* data, int height, int width)
    {
        if (!data || height <= 0 || width <= 0)
            return;

        // 1) Compute plane pointers
        size_t planeSize = static_cast<size_t>(height) * width;
        const uint16_t* ptrR = data + 0 * planeSize;
        const uint16_t* ptrG = data + 1 * planeSize;
        const uint16_t* ptrB = data + 2 * planeSize;

        // 2) Upload each U16 plane into a GpuMat
        cv::cuda::GpuMat R16_gpu(height, width, CV_16UC1);
        cv::cuda::GpuMat G16_gpu(height, width, CV_16UC1);
        cv::cuda::GpuMat B16_gpu(height, width, CV_16UC1);

        // Note: these temporary Mats wrap your host pointer but do NOT copy device data yet
        cv::Mat R16_host(height, width, CV_16UC1, const_cast<uint16_t*>(ptrR));
        cv::Mat G16_host(height, width, CV_16UC1, const_cast<uint16_t*>(ptrG));
        cv::Mat B16_host(height, width, CV_16UC1, const_cast<uint16_t*>(ptrB));

        R16_gpu.upload(R16_host);
        G16_gpu.upload(G16_host);
        B16_gpu.upload(B16_host);

        // 3) Scale 16U→8U on GPU
        //    Here we assume full 0–65535 range; adjust alpha if your data stays below that.
        const double alpha = 255.0 / 65535.0;
        cv::cuda::GpuMat R8_gpu, G8_gpu, B8_gpu;
        R16_gpu.convertTo(R8_gpu, CV_8UC1, alpha);
        G16_gpu.convertTo(G8_gpu, CV_8UC1, alpha);
        B16_gpu.convertTo(B8_gpu, CV_8UC1, alpha);

        // 4) Merge into BGR on GPU
        std::vector<cv::cuda::GpuMat> channels{ B8_gpu, G8_gpu, R8_gpu };
        cv::cuda::GpuMat bgr_gpu;
        cv::cuda::merge(channels, bgr_gpu);

        // 5) Download the final BGR8 image and display
        cv::Mat bgr_host;
        bgr_gpu.download(bgr_host);

        static const std::string winName = "CUDA: From LabVIEW (planar U16 RGB)";
        // Use WINDOW_NORMAL so you can resize by hand
        cv::namedWindow(winName, cv::WINDOW_NORMAL);
        cv::imshow(winName, bgr_host);
        cv::waitKey(1);
    }

    LABVIEWCUDAMEMORY_API int DemosaicRGGBNearest(
        const unsigned char* rawU8,
        unsigned long byteCount,
        int width,
        int height,
        float scaleR,
        float scaleG,
        float scaleB
    ) {
        //Start timing
        auto t0 = std::chrono::high_resolution_clock::now();

        // Sanity checks
        if (rawU8 == nullptr)                 return -1;  // null pointer
        if (byteCount == 0)                   return -2;  // no data at all

        // Apply defaults if caller passed zeros
        if (width <= 0) width = DEF_WIDTH;
        if (height <= 0) height = DEF_HEIGHT;
        if (scaleR <= 0) scaleR = DEF_R;
        if (scaleG <= 0) scaleG = DEF_G;
        if (scaleB <= 0) scaleB = DEF_B;

        // Compute payload size and strip header ---
        size_t pixelBytes = size_t(width) * size_t(height) * sizeof(uint16_t);
        if (byteCount < pixelBytes)          return -3;  // too few bytes for image
        size_t headerBytes = byteCount - pixelBytes;
        const uint16_t* rawBayer = nullptr;
        try {
            rawBayer = reinterpret_cast<const uint16_t*>(rawU8 + headerBytes);
        }
        catch (...) {
            return -4;  // pointer arithmetic failed (unlikely)
        }

        //Allocate and demosaic into internal buffer ---
        std::vector<unsigned char> rgbBuffer;
        try {
            rgbBuffer.resize(size_t(width) * size_t(height) * 3);
        }
        catch (const std::bad_alloc&) {
            return -5;  // memory allocation failure
        }

        for (int y = 0; y < height; ++y) {
            bool evenRow = ((y & 1) == 0);
            for (int x = 0; x < width; ++x) {
                bool evenCol = ((x & 1) == 0);
                size_t idx = size_t(y) * width + x;
                uint16_t r, g, b;

                if (evenRow && evenCol) {
                    r = rawBayer[idx];
                    g = rawBayer[y * width + clamp(x + 1, 0, width - 1)];
                    b = rawBayer[clamp(y + 1, 0, height - 1) * width + clamp(x + 1, 0, width - 1)];
                }
                else if (evenRow && !evenCol) {
                    g = rawBayer[idx];
                    r = rawBayer[y * width + clamp(x - 1, 0, width - 1)];
                    b = rawBayer[clamp(y + 1, 0, height - 1) * width + x];
                }
                else if (!evenRow && evenCol) {
                    g = rawBayer[idx];
                    r = rawBayer[clamp(y - 1, 0, height - 1) * width + x];
                    b = rawBayer[y * width + clamp(x + 1, 0, width - 1)];
                }
                else {
                    b = rawBayer[idx];
                    g = rawBayer[y * width + clamp(x - 1, 0, width - 1)];
                    r = rawBayer[clamp(y - 1, 0, height - 1) * width + clamp(x - 1, 0, width - 1)];
                }

                rgbBuffer[3 * idx + 0] = toU8(r * scaleR);
                rgbBuffer[3 * idx + 1] = toU8(g * scaleG);
                rgbBuffer[3 * idx + 2] = toU8(b * scaleB);
            }
        }

        // Display via OpenCV with timing overlay ---
        try {
            cv::Mat img(height, width, CV_8UC3, rgbBuffer.data());

            // Stop timing and compute metrics
            auto t1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = t1 - t0;
            double ms = elapsed.count();
            double fps = ms > 0.0 ? 1000.0 / ms : 0.0;

            // Draw overlay text
            char overlay[64];
            std::snprintf(overlay, sizeof(overlay),
                "Time: %.2f ms, FPS: %.1f", ms, fps);
            cv::putText(img,
                overlay,
                cv::Point(10, 50),           // moved lower
                cv::FONT_HERSHEY_SIMPLEX,
                1.4f,                        // size
                cv::Scalar(255, 255, 255),   // white text
                4,                           // thicker stroke
                cv::LINE_AA);                // anti-aliased

            // Create window once, then update
            static bool windowCreated = []() {
                cv::namedWindow("Demosaiced RGGB → RGB", cv::WINDOW_NORMAL);
                return true;
                }();

            cv::imshow("Demosaiced RGGB → RGB", img);
            cv::waitKey(1);  
        }
        catch (const cv::Exception&) {
            return -6;  // OpenCV display error
        }
        catch (...) {
            return -99; // unknown error
        }

        return 0;  // success
    }
}