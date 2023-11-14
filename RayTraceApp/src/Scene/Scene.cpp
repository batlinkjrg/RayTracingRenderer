#include "Scene.h";

void Scene::SetScene() {

	{
		SimpleSphereInfo sphereInfo;
		sphereInfo.material.color = { 0.0f, 0.5f, 1.0f };
		sphereInfo.position = { 0.0f, 0.0f, 0.0f };
		sphereInfo.radius = 0.5f;
		addSphere(sphereInfo); // Place Sphere
	}

	{
		SimpleSphereInfo sphereInfo;
		sphereInfo.material.color = { 1.0f, 0.0f, 1.0f };
		sphereInfo.position = { 1.0f, -1.0f, -5.0f };
		sphereInfo.radius = 1.0f;
		addSphere(sphereInfo); // Place Sphere
	}

}

SimpleSphereInfo* Scene::getScene() {
	createSphereBuffer();
	return sphereBufferCache;
}

SimpleSphereInfo Scene::createSphere() {
	SimpleSphereInfo sphere;

	// Set color
	float r = FloatRandomGen(0, 255, NON_NAGATIVE);
	float g = FloatRandomGen(0, 255, NON_NAGATIVE);
	float b = FloatRandomGen(0, 255, NON_NAGATIVE);
	sphere.material.color = { r, g, b };

	// Set posision
	float x = FloatRandomGen(0, 15, POSSIBLE_NAGATIVE);
	float y = FloatRandomGen(0, 15, POSSIBLE_NAGATIVE);
	float z = FloatRandomGen(0, 15, POSSIBLE_NAGATIVE);
	sphere.position = { x, y, z };

	// Sphere Size
	sphere.radius = FloatRandomGen(0, 1, NON_NAGATIVE);

	return sphere;
}

void Scene::addSphere(SimpleSphereInfo& sphere) {
	sphereSet.push_back(sphere);
	SphereCount++;
	SceneUpdate = true;
}

void Scene::removeSphere(int index) {
	if (SphereCount == 1) { return; }
	sphereSet.erase(sphereSet.begin() + index);
	SphereCount--;
	SceneUpdate = true;
}

void Scene::createSphereBuffer() {
	if (!SceneUpdate) { return; }

	// Check to see if the buffer is already empty
	if (sphereBufferCache != nullptr) {
		free(sphereBufferCache);
	}
	
	// Allocate buffer
	sphereBufferCache = (SimpleSphereInfo*)malloc(SphereCount * sizeof(SimpleSphereInfo));

	// Add spheres to buffer
	int index = 0;
	for each (SimpleSphereInfo sphere in sphereSet) {
		sphereBufferCache[index] = sphere;
		index++;
	}

	SceneUpdate = false;
}

// Utils
float Scene::FloatRandomGen(float min, float max, int sign) {
	// Check that min and max are correct
	if (min > max) {
		float buf = min;
		max = min;
		min = buf;
	}

	// Get a random number that is within range of min
	uint32_t seed = static_cast<uint32_t>(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
	seed = seed * 74779604 + 2891336453;
	uint32_t result = (seed >> (((seed >> 28) + 4)) ^ seed) * 277803737;
	result = (result >> 22) ^ result;
	float finalResult = result / 4294967295;
	
	// Make sure the float is in the correct range
	float randNum = ((max - min) * (((randNum) / (float) RAND_MAX)) + min);

	// Random chance of the float being negatvie
	if (!(rand() % 2) && !sign)
		return randNum * (float)(-1);
	else
		return randNum ;
}
