#pragma once

#include <glm/glm.hpp>

using namespace glm;

// Hold data for a ray
struct Ray {
	vec3 Origin;
	vec3 Direction;

	bool hit = false;
	float dst = FLT_MAX;
	uint32_t ObjectIndex;
};

// Store hit infor
struct HitInfo {
	// Ray Info
	bool didHit;
	float hitDistance;

	// Object Info
	uint32_t ObjectIndex;

	// World Info
	vec3 worldPos;
	vec3 worldNormal;
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
struct RenderInfo {
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
	
	// Random Number Seed
	uint32_t seed;

	// Ray Tracing Settings
	int BounceCount = 1;
};