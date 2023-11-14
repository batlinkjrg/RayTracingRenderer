#pragma once

#include <glm/glm.hpp>

using namespace glm;

// Hold data for a ray
struct Ray {
	vec3 Origin;
	vec3 Direction;
};

// Store hit infor
struct HitInfo {
	bool didHit;
	float dst;
	vec3 hitPoint;
	vec3 normal;
};

// Simple Color Info
struct SimpleMaterialInfo {
	vec3 color;
};

// Simple Sphere
struct SimpleSphereInfo {
	// Sphere Info
	bool visable = true;
	vec3 position;
	float radius;
	SimpleMaterialInfo material;
};

// Image Processing Info
struct PreRenderInfo {
	// Camera Info
	vec3 CameraPosition;
	mat4 CameraInverseView;
	mat4 CameraInverseProjection;
	
	// Image Info
	unsigned long long PixelBufferSize;
	int PixelCount;
	int ImageWidth;
	int ImageHeight;

	// Scene Info
	int SphereCount;
	int MaterialCount;
	
	uint32_t seed;
};