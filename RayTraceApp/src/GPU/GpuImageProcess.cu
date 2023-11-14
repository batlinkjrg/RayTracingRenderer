#include "GpuImageProcess.h"

__global__ void ImageGeneration(uint32_t* imageDataGPUptr, SimpleSphereInfo* gpuSphereBuffer, PreRenderInfo preRenderInfo) {

    // Use grid stepping method to loop through all the pixels
    // This allows for a flexible kernal launch, meaning all pixels are hit regardless of kernal luanch
    // This for loop will stop through all the pixels, whether that be all or grid-stepping //
    // 'i' is the index in the pixel array //
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < preRenderInfo.PixelCount; i += blockDim.x * gridDim.x) {

        // Execute the pixel shader // This will simulate using glsl! //
        vec2 coords = GetPixelCoordinatesFromIndex(i, preRenderInfo.ImageWidth, preRenderInfo.ImageHeight);

        Ray ray = CalculateRayDirection(coords, preRenderInfo.CameraPosition, preRenderInfo.CameraInverseView, preRenderInfo.CameraInverseProjection);

        vec4 color = TraceRay(ray, gpuSphereBuffer, preRenderInfo);
        color = clamp(color, vec4(0.0f), vec4(1.0f));

        imageDataGPUptr[i] = Vec4ToRGBA(color);
    }
}

__device__ vec4 TraceRay(const Ray& ray, SimpleSphereInfo* spheres, const PreRenderInfo& preRenderInfo) {

    // Gaurd Clause
    if (spheres == nullptr) { return vec4(0, 0, 0, 1); }
    
    // Prepare Some Hit Info
    SimpleSphereInfo* closestSphere = nullptr;
    float closestHit = FLT_MAX;

    for (int index = 0; index < preRenderInfo.SphereCount; index++) {

        // Get current sphere and immediatly check if its visable first //
        SimpleSphereInfo& sphere = spheres[index];
        if (!sphere.visable) { continue; }

        // Add a third dimension (depth) for the ray direction
        vec3 rayOrigin = ray.Origin - sphere.position;
        vec3 rayDirection = ray.Direction;
        rayDirection = normalize(rayDirection); // normalize the rayDirection

        // Variables
        float r = 0.5f;                                     // Radius of sphere

        float a = dot(rayDirection, rayDirection);          // Ray Origin
        float b = 2.0f * dot(rayOrigin, rayDirection);      // Ray Direction
        float c = dot(rayOrigin, rayOrigin) - (sphere.radius * sphere.radius);      // Ray Constant

        float discriminent = (b * b) - (4.0f * a * c);      // Inside the square root

        // If the ray misses exit early and move on to next sphere //
        if (discriminent < 0.0f) { continue; }

        // Smallest solution to the quadratic formula // Exact intersection point
        float hitDistance = ((-b) - sqrtf(discriminent)) / (2 * a);

        if (hitDistance < closestHit) { 
            closestHit = hitDistance; 
            closestSphere = &sphere; 
        }
    }

    if (closestSphere == nullptr) {
        return vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }


    // Add a third dimension (depth) for the ray direction
    vec3 rayOrigin = ray.Origin - closestSphere->position;
    vec3 rayDirection = ray.Direction;
    rayDirection = normalize(rayDirection); // normalize the rayDirection

    vec3 intersection;
    float lightIntensity;

    // Calculate the quadratic formula
    {

        intersection = normalize(rayOrigin + rayDirection * closestHit); // Normalize the hitpoint

        vec3 lightDirection(-1, -1, -1);
        lightDirection = normalize(lightDirection);

        // Invert Light Direction // Dot product gives us the cos of the angle between the values
        // If the value is nagetive than set it to zero //
        lightIntensity = max(dot(intersection, -lightDirection), 0.0f); // == cos(angle)
    }

    vec3 sphereColor = closestSphere->material.color;
    sphereColor += intersection * 0.5f + 0.5f;
    sphereColor *= lightIntensity;
    return vec4(sphereColor, 1);
}



// GPU Utilities //
__device__ uint32_t Vec4ToRGBA(const vec4& color) {
    // Set colors
    uint8_t a = (color.a * 255.0f);
    uint8_t r = (color.r * 255.0f);
    uint8_t g = (color.g * 255.0f);
    uint8_t b = (color.b * 255.0f);

    // Combine the colors
    return (a << 24) | (b << 16) | (g << 8) | r;
}

__device__ vec2 GetPixelCoordinatesFromIndex(int index, int width, int height) {
    // Get the current pixel location based off the width of the image //
    // i%width = x coord // floorf(i/width) = y coord //
    // Divide x by width and y by height to get the coordinates between 0 and 1 //
    // The vec2 coords are maped from 0 to 1 across the screen
    vec2 coords = { ((float)(index % width)) / width , ((float)floorf(index / width)) / height };
    // This will set the -1 to 1 instead of 0 to 1

    // Set aspect ratio // Aspect ratio = width/height //
    // coords.x *= (float)width / (float)height; // Camera Class currently does this for us

    // Set the space coordinates to between 0 and 1 instead of -1 and 1
    coords = coords * 2.0f - 1.0f;

    return coords;
}

__device__ Ray CalculateRayDirection(const vec2& coord, const vec3& rayOrigin, const mat4& inverseView, const mat4& inverseProjection) {
    Ray ray;
    ray.Origin = rayOrigin;

    vec4 target = inverseProjection * glm::vec4(coord.x, coord.y, 1, 1);
    vec3 rayDirection = glm::vec3(inverseView * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0)); // World space
    
    ray.Direction = rayDirection;

    return ray;
}

__device__ uint32_t UIntRandomGen(uint32_t seed) {
    seed = seed * 74779604 + 2891336453;
    uint32_t result = (seed >> (((seed >> 28) + 4)) ^ seed) * 277803737;
    result = (result >> 22) ^ result;
    return result;
}

__device__ float FloatRandomGen(uint32_t seed) {
    seed = seed * 74779604 + 2891336453;
    uint32_t result = (seed >> (((seed >> 28) + 4)) ^ seed) * 277803737;
    result = (result >> 22) ^ result;
    return result / 4294967295;
}

// Old Code //
// Per pixel shader //
__device__ vec4 PixelShader(vec2& coord) {

    // Add a third dimension (depth) for the ray direction
    vec3 rayOrigin(0.0f, 0.0f, 1.0f);
    vec3 rayDirection(coord.x, coord.y, -1.0f);
    rayDirection = normalize(rayDirection); // normalize the rayDirection

    /*
    *
    * Equation of a cirlce and equation of vector put into a quadratic equation
    * This allows us to use the quadratic formula
    *
    *      Value               Value                 Value
    *        a                   b                     c
    * (bx^2 + by^2)t^2 + (2(axbx + ayby))t + (ax^2 + ay^2 - r^2) = 0
    *
    * t = hit distance  // we want to solve for t
    * r = radius of cirle
    *
    * a = ray origin    // a = bx^2 + by^2 + bz^2    // a = ray direction * ray direction
    * b = ray direction // b = 2(axbx + ayby)     // b = ray direction * ray origin
    * c = ray constant  // c = (ax^2 + ay^2 - r^2) // c = (ray origin * ray origin) - radius^2
    *
    * Use quadratic equation to solve for t
    *
    * Quadratic formula
    * Plus or minus solutions -> s1 and s2 //
    *
    * -b +- sqrt(b^2 - 4ac)
    * ---------------------
    *         2a
    *
    * The descriminent of the quadratic formula is b^2 - 4ac // This is inside the square root
    * This can tell us if there was an intersection just not where
    *
    * If descriminent > 0
    * 2 Solutions // Hit
    *
    * If descriminent = 0
    * 1 Solution // Hit
    *
    * If descriminent < 0
    * 0 Solutions // No hit
    *
    */

    vec3 intersection;
    float lightIntensity;

    // Calculate the quadratic formula
    {
        // Variables
        float r = 0.5f;                                     // Radius of sphere

        float a = dot(rayDirection, rayDirection);          // Ray Origin
        float b = 2.0f * dot(rayOrigin, rayDirection);      // Ray Direction
        float c = dot(rayOrigin, rayOrigin) - (r * r);      // Ray Constant

        float discriminent = (b * b) - (4.0f * a * c);      // Inside the square root

        // If the ray misses exit early //
        if (discriminent < 0.0f) {
            return vec4(0, 0, 0, 1);
        }

        // Smallest solution to the quadratic formula // Exact intersection point
        float closetHit = ((-b) - sqrtf(discriminent)) / (2 * a);
        intersection = normalize(rayOrigin + rayDirection * closetHit); // Normalize the hitpoint

        vec3 lightDirection(-1, -1, -1);
        lightDirection = normalize(lightDirection);

        // Invert Light Direction // Dot product gives us the cos of the angle between the values
        // If the value is nagetive than set it to zero //
        lightIntensity = max(dot(intersection, -lightDirection), 0.0f); // == cos(angle)
    }

    vec3 sphereColor(1, 0.12f, 0.25f);
    sphereColor = intersection * 0.5f + 0.5f;
    sphereColor *= lightIntensity;
    return vec4(sphereColor, 1);
}


// Main Function //
void CudaImageGeneration(uint32_t* gpuPixelBuffer, SimpleSphereInfo* gpuSphereBuffer, PreRenderInfo& preRenderInfo) {

    // Use time since epoch as a seed for the random number generator
    preRenderInfo.seed = static_cast<uint32_t>(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());

    // Generate Image
    ImageGeneration <<< preRenderInfo.ImageWidth, 512 >>> (gpuPixelBuffer, gpuSphereBuffer ,preRenderInfo);
    cudaDeviceSynchronize();
}
