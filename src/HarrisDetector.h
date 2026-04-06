#pragma once
#include <vector>
#include "MathUtils.h"

struct KeyPoint {
    int x;
    int y;
    float response;
    
    KeyPoint(int _x, int _y, float _r) : x(_x), y(_y), response(_r) {}
};

class HarrisDetector {
public:
    // Convert 1-channel or 3-channel interleaved float matrix to 1-channel Matrix2D
    static MathUtils::Matrix2D ConvertToGrayMatrix(const unsigned char* data, int width, int height, int channels);

    // Compute both Harris and Lambda- responses at once to save time
    // Alternatively, expose two methods
    static std::vector<KeyPoint> DetectHarris(const MathUtils::Matrix2D& img, float k, float threshold, int minPoints, int nmsRadius = 1);
    static std::vector<KeyPoint> DetectLambdaMinus(const MathUtils::Matrix2D& img, float threshold, int minPoints, int nmsRadius = 1);

private:
    static void ComputeGradientsAndProducts(
        const MathUtils::Matrix2D& img,
        MathUtils::Matrix2D& Ix2, MathUtils::Matrix2D& Iy2, MathUtils::Matrix2D& Ixy);

    static std::vector<KeyPoint> NonMaximumSuppression(const MathUtils::Matrix2D& responseMap, float threshold, int radius);
};
