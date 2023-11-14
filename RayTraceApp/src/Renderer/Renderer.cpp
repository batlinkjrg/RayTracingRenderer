#include "Renderer.h"



void Renderer::RenderImage(Camera& camera, Scene& scene) {
	
	Timer timer;

	// Generate Image
	SetSphereBuffers(scene);
	SetPreRenderInfo(camera);
	CudaImageGeneration( gpuPixelBuffer, gpuSphereBuffer, preRenderInfo);
	SetRenderedImage();

	m_LastRenderTime = timer.ElapsedMillis();
	
}

void Renderer::SetPreRenderInfo(Camera& camera) {
	// Set cameraInfo
	preRenderInfo.CameraPosition = camera.GetPosition();
	preRenderInfo.CameraInverseView = camera.GetInverseView();
	preRenderInfo.CameraInverseProjection = camera.GetInverseProjection();

	// Set Image info
	preRenderInfo.ImageHeight = m_Image->GetHeight();
	preRenderInfo.ImageWidth = m_Image->GetWidth();
	preRenderInfo.PixelCount = pixelCount;
	preRenderInfo.PixelBufferSize = pixelBufferSize;

	// Set Scene Info
	preRenderInfo.SphereCount = sphereCount;
	
	// TODO: Add material buffer!!
}

// Set Rendered image to be displayed to screen
void Renderer::SetRenderedImage() {
	// Copy data from gpu, than to imageDataAddress than free it
	cudaError_t memSuccess = cudaMemcpy(m_ImageData, gpuPixelBuffer, pixelBufferSize, cudaMemcpyDeviceToHost);
	if (memSuccess != cudaSuccess) { std::cout << "Fatal Error: Can't copy pixel buffer from device!" << '\n'; }

	// Set image data to be rendered
	m_Image->SetData(m_ImageData);
}

// Resize the image
void Renderer::ResizeImage(uint32_t width, uint32_t height) {
	viewPort_h = height;
	viewPort_w = width;

	// Check to see if the image is created, otherwise just resize
	if (!m_Image) {
		m_Image = std::make_shared<Image>(width, height, ImageFormat::RGBA);
		SetPixelBuffers(width, height);
		return;
	}

	// Return if image is already the right size
	if (m_Image->GetWidth() == width && m_Image->GetHeight() == height)
		return;

	// Resize using current memory
	m_Image->Resize(width, height);
	SetPixelBuffers(width, height);
}

// Set pixel buffers
void Renderer::SetPixelBuffers(uint32_t& width, uint32_t& height) {
	// Delete Buffers
	delete[] m_ImageData;

	// Refresh total data points and memory size
	pixelCount = width * height;
	pixelBufferSize = (pixelCount) * sizeof(uint32_t);

	// Allocate Memory
	m_ImageData = (uint32_t*)malloc(pixelBufferSize);
	SetCudaBuffers();
}

// Set sphere buffers
void Renderer::SetSphereBuffers(Scene& scene) {
	// Set number of spheres and set buffer size
	sphereCount = scene.SphereTotal();
	sphereBufferSize = (sphereCount) * sizeof(SimpleSphereInfo);
	SetCudaBuffers();
	cudaMemcpy(gpuSphereBuffer, scene.getScene(), sphereBufferSize, cudaMemcpyHostToDevice);
}

// Check all cuda buffers and allocate or reallocate them as required
void Renderer::SetCudaBuffers() {
	if (old_pixelCount != pixelCount) {
		if (gpuPixelBuffer != nullptr) { FreeCudaBuffers(PixelBuffer); }
		cudaError_t cudaMemSuccess = cudaMalloc(&gpuPixelBuffer, pixelBufferSize);
		old_pixelCount = pixelCount;
		if (cudaMemSuccess != cudaSuccess) { std::cout << "Fatal Error: Can't allocate pixel buffer!" << '\n';  exit(-1); }
	}
	
	if (old_sphereCount != sphereCount) {
		if(gpuSphereBuffer != nullptr) { FreeCudaBuffers(SphereBuffer); }
		cudaError_t cudaMemSuccess = cudaMalloc(&gpuSphereBuffer, sphereBufferSize);
		old_sphereCount = sphereCount;
		if (cudaMemSuccess != cudaSuccess) { std::cout << "Fatal Error: Can't allocate sphere buffer!" << '\n';  exit(-1); }
	}
}

// Free cuda buffer of seleced type
void Renderer::FreeCudaBuffers(GpuBufferType type) {
	cudaError_t cudaMemSuccess;
	switch (type) {

		case PixelBuffer:
			cudaMemSuccess = cudaFree(gpuPixelBuffer);
			if (cudaMemSuccess != cudaSuccess) { std::cout << "Fatal Error: Failed to free pixel buffer!" << '\n';  exit(-1); }
			break;

		case SphereBuffer:
			cudaMemSuccess = cudaFree(gpuSphereBuffer);
			if (cudaMemSuccess != cudaSuccess) { std::cout << "Fatal Error: Failed to free sphere buffer!" << '\n'; exit(-1); }
			break;

		case MaterialBuffer:
			// TODO: Implement Material Buffer
			break;

	}
}


