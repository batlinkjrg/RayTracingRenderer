#include "GpuImageProcess.h"

__global__ void ImageGeneration(uint32_t* imageDataGPUptr, SimpleSphereInfo* gpuSphereBuffer, RenderInfo renderInfo) {

    // Use grid stepping method to loop through all the pixels
    // This allows for a flexible kernal launch, meaning all pixels are hit regardless of kernal luanch
    // This for loop will stop through all the pixels, whether that be all or grid-stepping //
    // 'i' is the index in the pixel array //
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < renderInfo.PixelCount; i += blockDim.x * gridDim.x) {
        if (gpuSphereBuffer == nullptr) { break; }

        // Execute the pixel shader // This will simulate using glsl! //
        vec2 coords = GetPixelCoordinatesFromIndex(i, renderInfo.ImageWidth, renderInfo.ImageHeight);

        // Execute Ray Tracing Pipeline
        vec4 color = RayGen(coords, gpuSphereBuffer, renderInfo);
        color = clamp(color, vec4(0.0f), vec4(1.0f));

        // Set Color
        imageDataGPUptr[i] = Vec4ToRGBA(color);
    }
}

__device__ vec4 RayGen(vec2 currentPixel, SimpleSphereInfo* spheres, RenderInfo& renderInfo) {
    // Create a ray
    Ray ray;
    CalculateRayDirection(ray, currentPixel, renderInfo);

    vec3 finalColor(0.0f);
    float shadingMultiplier = 1.0f;
    // Trace Ray for any number of bounces
    for (int i = 0; i < renderInfo.BounceCount; i++) {
        HitInfo hitInfo = TraceRay(ray, spheres, renderInfo);
        if (!hitInfo.didHit) { 
            vec3 backroundColor(0.0f, 0.0f, 0.0f);
            finalColor += backroundColor * shadingMultiplier;
            break;
        }

        // Set Light Direction
        vec3 lightDirection(-1, -1, -1);
        lightDirection = normalize(lightDirection);

        // If the value is nagetive than set it to zero //
        float lightIntensity = max(dot(hitInfo.worldNormal, -lightDirection), 0.0f); // == cos(angle)

        // Create a color
        vec3 sphereColor = spheres[ray.ObjectIndex].material.color;
        sphereColor *= lightIntensity;
        
        finalColor += sphereColor * shadingMultiplier;
        shadingMultiplier *= 0.6f;

        // Reset ray position // Move origin slightly off of sphere surface
        ray.Origin = hitInfo.worldPos + (hitInfo.worldNormal * 0.0001f);
        ray.Direction = reflect(ray.Direction, hitInfo.worldNormal);
    }
   


    return vec4(finalColor, 1);
}

__device__ HitInfo TraceRay(Ray& ray, SimpleSphereInfo* spheres, RenderInfo& renderInfo) {
    // Figure out if sphere was hit
    for (int index = 0; index < renderInfo.SphereCount; index++) {

        // Get current sphere and immediatly check if its visable first //
        SimpleSphereInfo& sphere = spheres[index];
        if (!sphere.visable) { continue; }

        // Origin of the ray, relative to the current sphere
        vec3 origin = ray.Origin - sphere.position;

        // Variables
        float a = dot(ray.Direction, ray.Direction);                          // Ray Origin
        float b = 2.0f * dot(origin, ray.Direction);                          // Ray Direction
        float c = dot(origin, origin) - (sphere.radius * sphere.radius);      // Ray Constant

        // Get descriminent, if less than zero no hit, next sphere
        float discriminent = (b * b) - (4.0f * a * c);     
        if (discriminent < 0.0f) { continue; }

        // Smallest solution to the quadratic formula // Exact intersection point // Test if closest point
        float hitDistance = ((-b) - sqrtf(discriminent)) / (2 * a);
        if (hitDistance > 0.0f && hitDistance < ray.dst) {
            ray.ObjectIndex = index;
            ray.dst = hitDistance;
            ray.hit = true;
        }
    }

    // If no spheres where hit than return a miss
    if (!ray.hit) {
        return Miss(ray);
    }

    return ClosestHit(ray, spheres);
}

__device__ HitInfo ClosestHit(Ray& ray, SimpleSphereInfo* spheres) {
    vec3 origin = ray.Origin - spheres[ray.ObjectIndex].position;

    // Create hit info
    HitInfo hitInfo;
    hitInfo.didHit = true;
    hitInfo.hitDistance = ray.dst;
    hitInfo.ObjectIndex = ray.ObjectIndex;
    hitInfo.worldPos = (origin + ray.Direction * ray.dst); // The ray hit point with sphere offset
    hitInfo.worldNormal = normalize(hitInfo.worldPos); // Normalize the hitpoint // Interestion

    // Remove sphere offset // Important to do after setting world normal
    hitInfo.worldPos += spheres[ray.ObjectIndex].position;
    
    return hitInfo;
}

__device__ HitInfo Miss(Ray& ray) {
    HitInfo hitInfo;
    hitInfo.didHit = false;
    return hitInfo;
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

__device__ void CalculateRayDirection(Ray& ray, const vec2& coord, RenderInfo& renderInfo) {
    ray.Origin = renderInfo.CameraPosition;

    vec4 target = renderInfo.CameraInverseProjection * glm::vec4(coord.x, coord.y, 1, 1);
    vec3 rayDirection = glm::vec3(renderInfo.CameraInverseView * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0)); // World space
    normalize(rayDirection);
    
    ray.Direction = rayDirection;
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


// Main Function //
void CudaImageGeneration(uint32_t* gpuPixelBuffer, SimpleSphereInfo* gpuSphereBuffer, RenderInfo& renderInfo) {
    // Generate Image
    ImageGeneration <<< renderInfo.ImageWidth, 512 >>> (gpuPixelBuffer, gpuSphereBuffer, renderInfo);
    cudaDeviceSynchronize();
}
