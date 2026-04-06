#pragma once
#include "MathUtils.h"
#include "HarrisDetector.h" // For KeyPoint
#include <vector>

struct SiftDescriptor {
    int x;
    int y;
    float scale;
    float orientation;
    std::vector<float> descriptor; // 128 elements

    SiftDescriptor() : x(0), y(0), scale(0.0f), orientation(0.0f) {}
};

class SiftDetector {
public:
    // Generate SIFT features from scratch given an image.
    static std::vector<SiftDescriptor> ExtractFeatures(const MathUtils::Matrix2D& img);
    
    // (Optional utility) Compute descriptor for provided keypoints (e.g. from Harris)
    static std::vector<SiftDescriptor> ExtractDescriptorsForPoints(const MathUtils::Matrix2D& img, const std::vector<KeyPoint>& keypoints);

private:
    static MathUtils::Matrix2D Downsample(const MathUtils::Matrix2D& img);
    
    // Extrema Detection
    // SIFT creates multiple octaves of multiple scales.
    
    // For simplicity in UI and runtime, this will be a somewhat reduced but fully custom implementation.
};
