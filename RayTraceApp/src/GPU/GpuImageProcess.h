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
__global__ void ImageGeneration(uint32_t* imageDataGPUptr, SimpleSphereInfo* gpuSphereBuffer, RenderInfo renderInfo);

// This will serve as the primary shader method
__device__ vec4 RayGen(vec2 currentPixel, SimpleSphereInfo* spheres, RenderInfo& preRenderInfo);

__device__ HitInfo TraceRay(Ray& ray, SimpleSphereInfo* spheres, RenderInfo& renderInfo);
__device__ HitInfo ClosestHit(Ray& ray, SimpleSphereInfo* spheres);
__device__ HitInfo Miss(Ray& ray);

// Convert Color Types
__device__ uint32_t Vec4ToRGBA(const vec4& color);

// Ray Tracing Functions
__device__ void CalculateRayDirection(Ray& ray, const vec2& coord, RenderInfo& renderInfo);

// Get pixel coordinates based off current index
__device__ vec2 GetPixelCoordinatesFromIndex(int index, int width, int height);

// Random number generators
__device__ float FloatRandomGen(uint32_t seed);
__device__ uint32_t UIntRandomGen(uint32_t seed);


// Helper function to be called in order to utilize gpu
void CudaImageGeneration(uint32_t* gpuPixelBuffer, SimpleSphereInfo* gpuSphereBuffer, RenderInfo& renderInfo);




