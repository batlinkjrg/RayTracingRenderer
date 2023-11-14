#pragma once

// STD Libs
#include <math.h>
#include <memory>
#include <chrono>
#include <ctime>

// Custom Libs
#include "../DataTypes.h"

// CUDA Libs
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// GLM Libs
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

using namespace glm;

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::system_clock;

// Create an image on the gpu
__global__ void ImageGeneration(uint32_t* imageDataGPUptr, SimpleSphereInfo* gpuSphereBuffer, PreRenderInfo preRenderInfo);

// This will serve as the primary shader method
__device__ vec4 TraceRay(const Ray& ray, SimpleSphereInfo* spheres, const PreRenderInfo& preRenderInfo);

// Convert Color Types
__device__ uint32_t Vec4ToRGBA(const vec4& color);

// Ray Tracing Functions
__device__ Ray CalculateRayDirection(const vec2& coord, const vec3& rayOrigin, const mat4& inverseView, const mat4& inverseProjection);

// Get pixel coordinates based off current index
__device__ vec2 GetPixelCoordinatesFromIndex(int index, int width, int height);

// Random number generators
__device__ float FloatRandomGen(uint32_t seed);
__device__ uint32_t UIntRandomGen(uint32_t seed);

// Old Code // Better Explination though //
[[deprecated("Replaced by TraceRay, which is more optimized")]]
__device__ vec4 PixelShader(vec2& coord);

// Helper function to be called in order to utilize gpu
void CudaImageGeneration(uint32_t* gpuPixelBuffer, SimpleSphereInfo* gpuSphereBuffer, PreRenderInfo& preRenderInfo);




