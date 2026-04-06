#include "SiftDetector.h"
#include <cmath>
#include <algorithm>

using namespace MathUtils;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::vector<SiftDescriptor> SiftDetector::ExtractFeatures(const Matrix2D& img) {
    // 1. Build a basic Gaussian pyramid (1 octave, 3 scales)
    float sigmas[4] = {1.6f, 2.015f, 2.539f, 3.2f};
    std::vector<Matrix2D> gaussians;
    
    for (int i = 0; i < 4; ++i) {
        gaussians.push_back(Convolve(img, GetGaussianKernel(sigmas[i])));
    }
    
    // 2. Build Difference of Gaussians (DoG)
    std::vector<Matrix2D> dogs;
    for (int i = 0; i < 3; ++i) {
        Matrix2D dog(img.width, img.height);
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                dog.at(x, y) = gaussians[i+1].at(x, y) - gaussians[i].at(x, y);
            }
        }
        dogs.push_back(dog);
    }
    
    // 3. Find Extrema in DoG (only the middle layer index 1 can have 3x3x3 neighbors)
    std::vector<KeyPoint> keypoints;
    float threshold = 8.0f; // Increased to 8.0 since pixels are 0-255 (filters low contrast noise)
    int gIndexForTesting = 1; // dogs[1]
    
    for (int y = 1; y < img.height - 1; ++y) {
        for (int x = 1; x < img.width - 1; ++x) {
            float val = dogs[1].at(x, y);
            if (std::abs(val) < threshold) continue;
            
            bool isMax = true;
            bool isMin = true;
            
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        float nVal = dogs[1 + dz].at(x + dx, y + dy);
                        if (val <= nVal) isMax = false;
                        if (val >= nVal) isMin = false;
                    }
                }
            }
            
            if (isMax || isMin) {
                keypoints.emplace_back(x, y, val);
            }
        }
    }
    
    // Extrema points are used for descriptor extraction
    return ExtractDescriptorsForPoints(gaussians[1], keypoints);
}

std::vector<SiftDescriptor> SiftDetector::ExtractDescriptorsForPoints(const Matrix2D& gaussianImg, const std::vector<KeyPoint>& keypoints) {
    std::vector<SiftDescriptor> descriptors;
    
    // Precompute gradients and orientations for the neighborhood
    Matrix2D mag(gaussianImg.width, gaussianImg.height);
    Matrix2D ori(gaussianImg.width, gaussianImg.height);
    
    for (int y = 1; y < gaussianImg.height - 1; ++y) {
        for (int x = 1; x < gaussianImg.width - 1; ++x) {
            float dx = gaussianImg.at(x + 1, y) - gaussianImg.at(x - 1, y);
            float dy = gaussianImg.at(x, y + 1) - gaussianImg.at(x, y - 1);
            mag.at(x, y) = std::sqrt(dx * dx + dy * dy);
            ori.at(x, y) = std::atan2(dy, dx);
        }
    }
    
    for (const auto& kp : keypoints) {
        int cx = kp.x;
        int cy = kp.y;
        
        // Ensure 16x16 window fits
        if (cx < 8 || cx >= gaussianImg.width - 8 || cy < 8 || cy >= gaussianImg.height - 8) {
            continue;
        }
        
        SiftDescriptor desc;
        desc.x = cx;
        desc.y = cy;
        desc.descriptor.resize(128, 0.0f);
        
        // Basic descriptor: 4x4 subregions, 8 orientation bins
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                int startY = cy - 8 + r * 4;
                int startX = cx - 8 + c * 4;
                
                std::vector<float> bins(8, 0.0f);
                
                for (int y = 0; y < 4; ++y) {
                    for (int x = 0; x < 4; ++x) {
                        float magnitude = mag.at(startX + x, startY + y);
                        float angle = ori.at(startX + x, startY + y);
                        
                        // Convert angle [-pi, pi] to [0, 2pi]
                        if (angle < 0) angle += 2.0f * M_PI;
                        
                        // Find bin
                        float binFloat = (angle / (2.0f * M_PI)) * 8.0f;
                        int bin = (int)(binFloat) % 8;
                        
                        // Add weight (we'd use gaussian weight here in full SIFT, simplified for basic)
                        bins[bin] += magnitude;
                    }
                }
                
                // Copy bins to vector
                int blockIdx = (r * 4 + c) * 8;
                for (int i = 0; i < 8; ++i) {
                    desc.descriptor[blockIdx + i] = bins[i];
                }
            }
        }
        
        // Normalize the descriptor to handle illumination changes
        float normSq = 0.0f;
        for (float v : desc.descriptor) {
            normSq += v * v;
        }
        float norm = std::sqrt(normSq);
        if (norm > 1e-6f) {
            for (float& v : desc.descriptor) {
                v /= norm;
                // Cap to 0.2
                if (v > 0.2f) v = 0.2f;
            }
            // Renormalize
            normSq = 0.0f;
            for (float v : desc.descriptor) {
                normSq += v * v;
            }
            norm = std::sqrt(normSq);
            for (float& v : desc.descriptor) {
                v /= norm;
            }
        }
        
        descriptors.push_back(desc);
    }
    
    return descriptors;
}
