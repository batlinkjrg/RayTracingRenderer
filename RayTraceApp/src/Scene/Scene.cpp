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
	float red = FloatRandomGen(0, 1.0f, NON_NAGATIVE);
	float green = FloatRandomGen(0, 1.0f, NON_NAGATIVE);
	float blue = FloatRandomGen(0, 1.0f, NON_NAGATIVE);
	sphere.material.color = { red, blue, green };

	// Set posision
	float boxDim = 10.0f;
	float x = FloatRandomGen(0, boxDim, POSSIBLE_NAGATIVE);
	float y = FloatRandomGen(0, boxDim, POSSIBLE_NAGATIVE);
	float z = FloatRandomGen(0, boxDim, POSSIBLE_NAGATIVE);
	sphere.position = { x, y, z };

	// Sphere Size
	sphere.radius = FloatRandomGen(0.10f, 1.0f, NON_NAGATIVE);

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
	
	// Make sure the float is in the correct range
	// Value = ( minimumValue + randomGeneration ) / ( maximumFloat / ( minimumValue - maximumValue ) )
	float randNum = min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max - min)));

	// Random chance of the float being negatvie
	if (sign == 0 && !(rand() % 2))
		return randNum * (float)(-1);
	else
		return randNum;
}
