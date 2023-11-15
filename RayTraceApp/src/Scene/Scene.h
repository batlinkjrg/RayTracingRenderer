#pragma once

#include "../DataTypes.h"

// STD Libs
#include <math.h>
#include <vector>
#include <chrono>
#include <ctime>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::system_clock;

enum randType { POSSIBLE_NAGATIVE = 0, NON_NAGATIVE = 1 };

class Scene {
public:

	// Sphere Scene Set
	std::vector<SimpleSphereInfo> sphereSet;

	// SphereManger // Sphere Helper Methods
	int SphereTotal() { return SphereCount; }
	static SimpleSphereInfo createSphere();
	void addSphere(SimpleSphereInfo& sphere);
	void removeSphere(int index);
	void updateScene() { SceneUpdate = true; }

	// Get Scene
	void SetScene();
	SimpleSphereInfo* getScene();
	

private:

	// Sphre count
	int SphereCount = 0;
	bool SceneUpdate = true;

	// Sphere Buffer cache
	SimpleSphereInfo* sphereBufferCache = nullptr;

	// Sphere Helper Methods
	void createSphereBuffer();

	// Utils
	static float FloatRandomGen(float min, float max, int sign);
};