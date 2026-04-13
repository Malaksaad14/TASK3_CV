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

struct MatchPair {
    int idx1;
    int idx2;
    float score;

    MatchPair() : idx1(-1), idx2(-1), score(0.0f) {}
    MatchPair(int i1, int i2, float s) : idx1(i1), idx2(i2), score(s) {}
};

class SiftDetector {
public:
    // Generate SIFT features from scratch given an image.
    static std::vector<SiftDescriptor> ExtractFeatures(const MathUtils::Matrix2D& img);
    
    // (Optional utility) Compute descriptor for provided keypoints (e.g. from Harris)
    static std::vector<SiftDescriptor> ExtractDescriptorsForPoints(const MathUtils::Matrix2D& img, const std::vector<KeyPoint>& keypoints);
    
    // Match descriptor sets using nearest-neighbor SSD (lower is better).
    static std::vector<MatchPair> MatchDescriptorsSSD(const std::vector<SiftDescriptor>& set1, const std::vector<SiftDescriptor>& set2);
    
    // Match descriptor sets using nearest-neighbor normalized cross correlation (higher is better).
    static std::vector<MatchPair> MatchDescriptorsNCC(const std::vector<SiftDescriptor>& set1, const std::vector<SiftDescriptor>& set2);

private:
    static MathUtils::Matrix2D Downsample(const MathUtils::Matrix2D& img);
    
    // Extrema Detection
    // SIFT creates multiple octaves of multiple scales.
    
    // For simplicity in UI and runtime, this will be a somewhat reduced but fully custom implementation.
};
