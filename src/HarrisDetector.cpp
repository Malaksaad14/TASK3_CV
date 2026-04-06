#include "HarrisDetector.h"
#include <algorithm>
#include <cmath>

using namespace MathUtils;

Matrix2D HarrisDetector::ConvertToGrayMatrix(const unsigned char* data, int width, int height, int channels) {
    Matrix2D gray(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            if (channels == 3) {
                // simple luminance BGR mapping (OpenCV is BGR)
                gray.at(x, y) = 0.114f * data[idx] + 0.587f * data[idx+1] + 0.299f * data[idx+2];
            } else if (channels == 1) {
                gray.at(x, y) = (float)data[idx];
            } else if (channels == 4) {
                gray.at(x, y) = 0.114f * data[idx] + 0.587f * data[idx+1] + 0.299f * data[idx+2];
            }
        }
    }
    return gray;
}

void HarrisDetector::ComputeGradientsAndProducts(
    const Matrix2D& img,
    Matrix2D& Ix2, Matrix2D& Iy2, Matrix2D& Ixy) {
    
    Matrix2D sobelX = GetSobelX();
    Matrix2D sobelY = GetSobelY();
    
    Matrix2D Ix = Convolve(img, sobelX, PaddingMode::REPLICATE);
    Matrix2D Iy = Convolve(img, sobelY, PaddingMode::REPLICATE);
    
    Ix2 = Matrix2D(img.width, img.height);
    Iy2 = Matrix2D(img.width, img.height);
    Ixy = Matrix2D(img.width, img.height);
    
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            float ix = Ix.at(x, y);
            float iy = Iy.at(x, y);
            Ix2.at(x, y) = ix * ix;
            Iy2.at(x, y) = iy * iy;
            Ixy.at(x, y) = ix * iy;
        }
    }
}

std::vector<KeyPoint> HarrisDetector::DetectHarris(const Matrix2D& img, float k, float threshold, int minPoints, int nmsRadius) {
    Matrix2D Ix2, Iy2, Ixy;
    ComputeGradientsAndProducts(img, Ix2, Iy2, Ixy);
    
    // Apply Gaussian blur (sigma = 1.0 or 1.5)
    Matrix2D gaussKernel = GetGaussianKernel(1.0f);
    Matrix2D Sxx = Convolve(Ix2, gaussKernel);
    Matrix2D Syy = Convolve(Iy2, gaussKernel);
    Matrix2D Sxy = Convolve(Ixy, gaussKernel);
    
    Matrix2D responseMap(img.width, img.height);
    
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            float sxx = Sxx.at(x, y);
            float syy = Syy.at(x, y);
            float sxy = Sxy.at(x, y);
            
            float det = sxx * syy - sxy * sxy;
            float trace = sxx + syy;
            float r = det - k * trace * trace;
            
            responseMap.at(x, y) = r;
        }
    }
    
    return NonMaximumSuppression(responseMap, threshold, nmsRadius);
}

std::vector<KeyPoint> HarrisDetector::DetectLambdaMinus(const Matrix2D& img, float threshold, int minPoints, int nmsRadius) {
    Matrix2D Ix2, Iy2, Ixy;
    ComputeGradientsAndProducts(img, Ix2, Iy2, Ixy);
    
    Matrix2D gaussKernel = GetGaussianKernel(1.0f);
    Matrix2D Sxx = Convolve(Ix2, gaussKernel);
    Matrix2D Syy = Convolve(Iy2, gaussKernel);
    Matrix2D Sxy = Convolve(Ixy, gaussKernel);
    
    Matrix2D responseMap(img.width, img.height);
    
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            float sxx = Sxx.at(x, y);
            float syy = Syy.at(x, y);
            float sxy = Sxy.at(x, y);
            
            // Eigenvalues of matrix:
            // [ sxx  sxy ]
            // [ sxy  syy ]
            // Characteristic equation: lambda^2 - (sxx+syy)lambda + (sxx*syy - sxy^2) = 0
            float trace = sxx + syy;
            float det = sxx * syy - sxy * sxy;
            
            // lambda_m = (trace - sqrt(trace^2 - 4*det)) / 2
            float inner = trace * trace - 4 * det;
            float lambda_min = 0.0f;
            if (inner >= 0.0f) {
                lambda_min = (trace - std::sqrt(inner)) / 2.0f;
            }
            
            responseMap.at(x, y) = lambda_min;
        }
    }
    
    return NonMaximumSuppression(responseMap, threshold, nmsRadius);
}

std::vector<KeyPoint> HarrisDetector::NonMaximumSuppression(const Matrix2D& responseMap, float threshold, int radius) {
    std::vector<KeyPoint> keypoints;
    
    for (int y = radius; y < responseMap.height - radius; ++y) {
        for (int x = radius; x < responseMap.width - radius; ++x) {
            float val = responseMap.at(x, y);
            if (val <= threshold) continue;
            
            bool isMax = true;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    
                    if (responseMap.at(x + dx, y + dy) >= val) {
                        isMax = false;
                        break;
                    }
                }
                if (!isMax) break;
            }
            
            if (isMax) {
                keypoints.emplace_back(x, y, val);
            }
        }
    }
    
    // Sort by response
    std::sort(keypoints.begin(), keypoints.end(), [](const KeyPoint& a, const KeyPoint& b) {
        return a.response > b.response;
    });
    
    return keypoints;
} 
