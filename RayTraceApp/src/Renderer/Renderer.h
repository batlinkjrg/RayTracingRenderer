#pragma once

// Walnut Libs
#include "Walnut/Image.h"
#include "Walnut/Timer.h"

// Standard Libs
#include <iostream>
#include <memory>

// Custom Libs
#include "../DataTypes.h"
#include "../Scene/Scene.h"
#include "../GPU/GpuImageProcess.h"
#include "../Camera/GpuCamera.h"

using namespace Walnut;

enum GpuBufferType { PixelBuffer, SphereBuffer, MaterialBuffer };

class Renderer {
public:
	Renderer() = default;

	std::shared_ptr<Image> m_Image;
	float m_LastRenderTime = 0.0f;

	void RenderImage(Camera& camera, Scene& scene);
	void ResizeImage(uint32_t width, uint32_t height);

	int getCurrentHeight() { return m_Image->GetHeight(); }
	int getCurrentWidth() { return m_Image->GetWidth(); }
	int getCurrentTotalPixels() { return pixelCount; }

	int* getCurrentBounceCount() { return &renderInfo.BounceCount; }

private:
	
	// Image data
	uint32_t* m_ImageData = nullptr;
	uint32_t viewPort_w = 0, viewPort_h = 0;

	// Items in buffers
	int pixelCount = 0;
	int old_pixelCount = 0;

	int sphereCount = 0;
	int old_sphereCount = 0;

	// Buffer Sizes
	unsigned long long pixelBufferSize = 0;
	unsigned long long sphereBufferSize = 0;

	// GPUData
	uint32_t* gpuPixelBuffer = nullptr;
	SimpleSphereInfo* gpuSphereBuffer = nullptr;

	// ImageProccessing Data
	RenderInfo renderInfo;

	void SetPreRenderInfo(Camera& camera);
	void SetRenderedImage();

	void SetPixelBuffers(uint32_t& width, uint32_t& height);
	void SetSphereBuffers(Scene& scene);
	void SetCudaBuffers();
	void FreeCudaBuffers(GpuBufferType type);
};
