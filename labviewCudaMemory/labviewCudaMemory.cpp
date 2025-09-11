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

    LABVIEWCUDAMEMORY_API void ShowLabVIEWRGB(uint64_t* LVImagePointer, int lineWidth, int height, int width)
    {
        if (!LVImagePointer || height <= 0 || width <= 0) return;

        // Directly cast to void* for OpenCV
        cv::Mat rgb(height, width, CV_8UC3, reinterpret_cast<void*>(LVImagePointer), lineWidth);

        // Fix orientation: transpose + flip
        cv::Mat rgbFixed;
        cv::transpose(rgb, rgbFixed);
        cv::flip(rgbFixed, rgbFixed, 1);

        cv::Mat bgr;
        cv::cvtColor(rgbFixed, bgr, cv::COLOR_RGB2BGR);

        cv::imshow("LabVIEW RGB", bgr);
        cv::waitKey(1);
    }

    LABVIEWCUDAMEMORY_API void ShowImageFromInterleavedRGB_U16(const uint16_t* data, int height, int width) {
        if (!data || height <= 0 || width <= 0) return;
        cv::Mat rgb16(height, width, CV_16UC3, const_cast<uint16_t*>(data));
        cv::Mat bgr16;        
        cv::cvtColor(rgb16, bgr16, cv::COLOR_RGB2BGR);
        const double scale = 1.0 / 256.0;
        cv::Mat bgr8;
        bgr16.convertTo(bgr8, CV_8UC3, scale);

        // Show image (non-blocking update)
        cv::imshow("From LabVIEW (interleaved U16 RGB)", bgr8);
        cv::waitKey(1);
    }

    LABVIEWCUDAMEMORY_API void ShowImageFromPlanarU16(uint64_t* LVImagePointer, int lineWidth, int height, int width)
    {
        if (!LVImagePointer || height <= 0 || width <= 0 || lineWidth < width * 2) return;

        uint8_t* base = reinterpret_cast<uint8_t*>(LVImagePointer);
        int rowBytes = lineWidth;

        cv::Mat r(height, width, CV_16UC1, reinterpret_cast<void*>(base), rowBytes);
        cv::Mat g(height, width, CV_16UC1, reinterpret_cast<void*>(base + rowBytes * height), rowBytes);
        cv::Mat b(height, width, CV_16UC1, reinterpret_cast<void*>(base + 2 * rowBytes * height), rowBytes);

        cv::Mat rgb16;
        cv::merge(std::vector<cv::Mat>{r, g, b}, rgb16);

        cv::Mat rgb8;
        rgb16.convertTo(rgb8, CV_8UC3, 1.0 / 256.0);

        cv::Mat rgbFixed;
        cv::transpose(rgb8, rgbFixed);
        cv::flip(rgbFixed, rgbFixed, 1);

        cv::Mat bgr;
        cv::cvtColor(rgbFixed, bgr, cv::COLOR_RGB2BGR);
        cv::imshow("Planar U16 RGB", bgr);
        cv::waitKey(1);
    }
    
    LABVIEWCUDAMEMORY_API void ShowImageFromInterleavedU16(const uint16_t* data, int height, int width)
    {
        if (!data || height <= 0 || width <= 0) return;

        cv::Mat rgb16(height, width, CV_16UC3, (void*)data);

        // Convert to 8-bit for display
        cv::Mat rgb8;
        rgb16.convertTo(rgb8, CV_8UC3, 1.0 / 256.0);

        cv::Mat bgr;
        cv::cvtColor(rgb8, bgr, cv::COLOR_RGB2BGR);

        cv::imshow("Interleaved U16 RGB", bgr);
        cv::waitKey(1);
    }

    //LABVIEWCUDAMEMORY_API void ShowImageFromPlanarU16_LW(const uint16_t* data, int height, int width, int lineWidth)
    //{
    //    if (!data || height <= 0 || width <= 0 || lineWidth < width * 2) return;

    //    const int planeBytes = lineWidth * height;
    //    const uint8_t* base = reinterpret_cast<const uint8_t*>(data);

    //    cv::Mat r(height, width, CV_16UC1, (void*)base, lineWidth);
    //    cv::Mat g(height, width, CV_16UC1, (void*)(base + planeBytes), lineWidth);
    //    cv::Mat b(height, width, CV_16UC1, (void*)(base + 2 * planeBytes), lineWidth);

    //    cv::Mat rgb16;
    //    cv::merge(std::vector<cv::Mat>{r, g, b}, rgb16);

    //    // Convert to 8-bit for display
    //    cv::Mat rgb8;
    //    rgb16.convertTo(rgb8, CV_8UC3, 1.0 / 256.0);

    //    cv::Mat bgr;
    //    cv::cvtColor(rgb8, bgr, cv::COLOR_RGB2BGR);

    //    cv::imshow("Planar U16 RGB", bgr);
    //    cv::waitKey(1);
    //}

    //LABVIEWCUDAMEMORY_API void ShowImageFromInterleavedU16_LW(const uint16_t* data, int height, int width, int lineWidth)
    //{
    //    if (!data || height <= 0 || width <= 0 || lineWidth < width * 3 * 2) return;

    //    // Each row has 'lineWidth' bytes
    //    cv::Mat rgb16(height, width, CV_16UC3, (void*)data, lineWidth);

    //    // Convert to 8-bit for display
    //    cv::Mat rgb8;
    //    rgb16.convertTo(rgb8, CV_8UC3, 1.0 / 256.0);

    //    cv::Mat bgr;
    //    cv::cvtColor(rgb8, bgr, cv::COLOR_RGB2BGR);

    //    cv::imshow("Interleaved U16 RGB", bgr);
    //    cv::waitKey(1);
    //}

    LABVIEWCUDAMEMORY_API void ShowImageFromInterleavedU16_LW(const uint16_t* data, int height, int width, int lineWidth)
    {
        if (!data || height <= 0 || width <= 0 || lineWidth < width * 3 * 2) return;

        cv::Mat rgb16(height, width, CV_16UC3, (void*)data, lineWidth);

        // Convert to 8-bit for display
        cv::Mat rgb8;
        rgb16.convertTo(rgb8, CV_8UC3, 1.0 / 256.0);

        // Fix orientation: transpose + flip
        cv::Mat rgbFixed;
        cv::transpose(rgb8, rgbFixed);
        cv::flip(rgbFixed, rgbFixed, 1);

        cv::Mat bgr;
        cv::cvtColor(rgbFixed, bgr, cv::COLOR_RGB2BGR);
        cv::imshow("Interleaved U16 RGB", bgr);
        cv::waitKey(1);
    }

    LABVIEWCUDAMEMORY_API void ShowImageFromPlanarU16_LW(const uint16_t* data, int height, int width, int lineWidth)
    {
        if (!data || height <= 0 || width <= 0 || lineWidth < width * 2) return;

        const uint8_t* base = reinterpret_cast<const uint8_t*>(data);
        int rowBytes = lineWidth;

        cv::Mat r(height, width, CV_16UC1, (void*)base, rowBytes);
        cv::Mat g(height, width, CV_16UC1, (void*)(base + rowBytes * height), rowBytes);
        cv::Mat b(height, width, CV_16UC1, (void*)(base + 2 * rowBytes * height), rowBytes);

        cv::Mat rgb16;
        cv::merge(std::vector<cv::Mat>{r, g, b}, rgb16);

        // Convert to 8-bit for display
        cv::Mat rgb8;
        rgb16.convertTo(rgb8, CV_8UC3, 1.0 / 256.0);

        // Fix orientation: transpose + flip
        cv::Mat rgbFixed;
        cv::transpose(rgb8, rgbFixed);
        cv::flip(rgbFixed, rgbFixed, 1);

        cv::Mat bgr;
        cv::cvtColor(rgbFixed, bgr, cv::COLOR_RGB2BGR);
        cv::imshow("Planar U16 RGB", bgr);
        cv::waitKey(1);
    }

    LABVIEWCUDAMEMORY_API void ShowImageAutoFix(const void* buffer, int dim0,  int dim1, int dim2, int elemBytes) 
    {
        if (!buffer || dim0 <= 0 || dim1 <= 0 || dim2 <= 0) return;
        if (elemBytes != 1 && elemBytes != 2) return;

        const uint8_t* raw = static_cast<const uint8_t*>(buffer);

        struct Candidate {
            cv::Mat img;    // interleaved CV_8UC3 or CV_8UC4
            std::string desc;
            double score;
        };
        std::vector<Candidate> candidates;

        auto score_image = [](const cv::Mat& im)->double {
            if (im.empty()) return -1.0;
            cv::Mat gray;
            if (im.channels() == 3) cv::cvtColor(im, gray, cv::COLOR_BGR2GRAY);
            else if (im.channels() == 4) cv::cvtColor(im, gray, cv::COLOR_BGRA2GRAY);
            else gray = im;

            // compute mean abs diff between adjacent rows
            cv::Mat diff;
            cv::absdiff(gray.rowRange(0, gray.rows - 1), gray.rowRange(1, gray.rows), diff);
            if (diff.empty()) return -1.0;
            return cv::mean(diff)[0]; // scalar mean
            };

        auto try_interleaved = [&](int H, int W, int channels, const void* ptr)->cv::Mat {
            if (H <= 0 || W <= 0) return cv::Mat();
            int type = (elemBytes == 1) ? (CV_8UC(channels)) : (CV_16UC(channels));
            size_t step = static_cast<size_t>(W) * channels * elemBytes; // bytes/row, assumes no padding
            cv::Mat src(H, W, type, const_cast<void*>(ptr), step);

            cv::Mat converted;
            if (elemBytes == 2) {
                // convert 16-bit (assumed big range) to 8-bit for display
                src.convertTo(converted, CV_8UC(channels), 1.0 / 256.0);
            }
            else converted = src;

            // If RGB order, convert to BGR for imshow
            if (channels == 3) {
                cv::Mat bgr;
                cv::cvtColor(converted, bgr, cv::COLOR_RGB2BGR);
                return bgr;
            }
            else if (channels == 4) {
                cv::Mat bgr;
                cv::cvtColor(converted, bgr, cv::COLOR_RGBA2BGR);
                return bgr;
            }
            else {
                return converted;
            }
            };

        auto try_planar = [&](int C, int H, int W, const void* ptr)->cv::Mat {
            if (C < 3 || H <= 0 || W <= 0) return cv::Mat();
            size_t planeBytes = static_cast<size_t>(H) * static_cast<size_t>(W) * elemBytes;
            // pointers to each plane
            const void* p0 = ptr;
            const void* p1 = static_cast<const uint8_t*>(ptr) + planeBytes;
            const void* p2 = static_cast<const uint8_t*>(ptr) + planeBytes * 2;

            int type1 = (elemBytes == 1) ? CV_8UC1 : CV_16UC1;
            cv::Mat ch0(H, W, type1, const_cast<void*>(p0));
            cv::Mat ch1(H, W, type1, const_cast<void*>(p1));
            cv::Mat ch2(H, W, type1, const_cast<void*>(p2));
            std::vector<cv::Mat> chans = { ch0, ch1, ch2 };
            cv::Mat merged;
            cv::merge(chans, merged); // CV_8UC3 or CV_16UC3

            cv::Mat converted;
            if (elemBytes == 2) merged.convertTo(converted, CV_8UC3, 1.0 / 256.0);
            else converted = merged;

            cv::Mat bgr;
            cv::cvtColor(converted, bgr, cv::COLOR_RGB2BGR);
            return bgr;
            };

        // Interpretations to try:
        // From the LabVIEW 3D array, we don't know which axis is channels vs H vs W.
        // Typical layouts: [C,H,W] or [H,W,C]. We will try both, with channels=3 or 4,
        // and with H/W swapped.
        std::vector<std::tuple<int, int, int, std::string>> tries;

        // assume dim0 is C
        tries.emplace_back(dim0, dim1, dim2, "C,H,W (interpreted as planar: C x H x W)");
        // assume dim2 is C (common H x W x C)
        tries.emplace_back(dim1, dim2, dim0, "H,W,C (interpreted as interleaved H x W x C)");
        // assume dims swapped (maybe user passed width/height reversed)
        tries.emplace_back(dim0, dim2, dim1, "C,W,H (planar with swapped H/W)");
        tries.emplace_back(dim2, dim1, dim0, "W,H,C (interleaved with swapped dims)");

        // For each try, attempt both planar (if first element==3) and interleaved
        for (auto& t : tries) {
            int a = std::get<0>(t), b = std::get<1>(t), c = std::get<2>(t);
            std::string desc = std::get<3>(t);

            // If first interpreted as channels (small number) and equals 3 or 4, try planar
            if (a == 3 || a == 4) {
                // planar: C x H x W  -> channels=a, H=b, W=c
                const void* ptr = raw;
                // sanity check: buffer size must be at least channels*H*W*elemBytes
                size_t needed = static_cast<size_t>(a) * static_cast<size_t>(b) * static_cast<size_t>(c) * elemBytes;
                // We don't have actual size information, so skip if obviously insane
                if (needed > (static_cast<size_t>(1) << 34)) { /*skip*/ }
                else {
                    cv::Mat img = try_planar(a, b, c, ptr);
                    if (!img.empty()) {
                        double sc = score_image(img);
                        candidates.push_back({ img, desc + " [planar]", sc });
                    }
                }
            }

            // Interleaved: assume H=a, W=b, C=c (if c==3 or 4)
            if (c == 3 || c == 4) {
                const void* ptr = raw;
                cv::Mat img = try_interleaved(a, b, c, ptr);
                if (!img.empty()) {
                    double sc = score_image(img);
                    candidates.push_back({ img, desc + " [interleaved]", sc });
                }
            }
        }

        // Also try simple interleaved with (height=dim1,width=dim2,channels=3) as common case
        {
            cv::Mat img = try_interleaved(dim1, dim2, 3, raw);
            if (!img.empty()) candidates.push_back({ img, "Common H=dim1 W=dim2 C=3 [interleaved]", score_image(img) });
            // try swapped dims
            img = try_interleaved(dim2, dim1, 3, raw);
            if (!img.empty()) candidates.push_back({ img, "Swapped H/W with C=3 [interleaved]", score_image(img) });
        }

        // pick best candidate by score
        double bestScore = -std::numeric_limits<double>::infinity();
        int bestIdx = -1;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i].score > bestScore) {
                bestScore = candidates[i].score;
                bestIdx = static_cast<int>(i);
            }
        }

        if (bestIdx >= 0) {
            const Candidate& c = candidates[bestIdx];
            std::cerr << "[ShowImageAutoFix] chosen: " << c.desc << "  score=" << c.score << "\n";
            cv::imshow("From LabVIEW (auto)", c.img);
            cv::waitKey(1);
        }
        else {
            std::cerr << "[ShowImageAutoFix] No valid interpretation produced a non-empty image.\n";
        }
    } 

   
}