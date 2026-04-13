#include "SiftDetector.h"
#include <cmath>
#include <algorithm>
#include <limits>

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
    float threshold = 4.0f; // Decreased from 8.0 to allow more descriptors to pass
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

std::vector<MatchPair> SiftDetector::MatchDescriptorsSSD(const std::vector<SiftDescriptor>& set1, const std::vector<SiftDescriptor>& set2) {
    std::vector<MatchPair> matches;
    if (set1.empty() || set2.empty()) {
        return matches;
    }

    matches.reserve(set1.size());
    for (size_t i = 0; i < set1.size(); ++i) {
        const auto& d1 = set1[i].descriptor;
        float bestDist = std::numeric_limits<float>::max();
        int bestIdx = -1;

        for (size_t j = 0; j < set2.size(); ++j) {
            const auto& d2 = set2[j].descriptor;
            if (d1.size() != d2.size() || d1.empty()) continue;

            float dot = 0.0f;
            for (size_t k = 0; k < d1.size(); ++k) {
                dot += d1[k] * d2[k];
            }
            float ssd = 2.0f - 2.0f * dot;

            if (ssd < bestDist) {
                bestDist = ssd;
                bestIdx = static_cast<int>(j);
            }
        }

        if (bestIdx >= 0) {
            matches.emplace_back(static_cast<int>(i), bestIdx, bestDist);
        }
    }

    return matches;
}

std::vector<MatchPair> SiftDetector::MatchDescriptorsNCC(const std::vector<SiftDescriptor>& set1, const std::vector<SiftDescriptor>& set2) {
    std::vector<MatchPair> matches;
    if (set1.empty() || set2.empty()) {
        return matches;
    }

    struct CenteredDescriptor {
        std::vector<float> centered;
        float invNorm;
    };

    std::vector<CenteredDescriptor> centered1(set1.size());
    std::vector<CenteredDescriptor> centered2(set2.size());

    auto prepareCentered = [](const SiftDescriptor& src, CenteredDescriptor& out) {
        const auto& d = src.descriptor;
        out.centered.assign(d.size(), 0.0f);
        if (d.empty()) {
            out.invNorm = 0.0f;
            return;
        }

        float mean = 0.0f;
        for (float v : d) mean += v;
        mean /= static_cast<float>(d.size());

        float normSq = 0.0f;
        for (size_t i = 0; i < d.size(); ++i) {
            float c = d[i] - mean;
            out.centered[i] = c;
            normSq += c * c;
        }
        out.invNorm = (normSq > 1e-8f) ? (1.0f / std::sqrt(normSq)) : 0.0f;
    };

    for (size_t i = 0; i < set1.size(); ++i) prepareCentered(set1[i], centered1[i]);
    for (size_t i = 0; i < set2.size(); ++i) prepareCentered(set2[i], centered2[i]);

    matches.reserve(set1.size());
    for (size_t i = 0; i < set1.size(); ++i) {
        const auto& c1 = centered1[i];
        float bestCorr = -std::numeric_limits<float>::max();
        int bestIdx = -1;

        for (size_t j = 0; j < set2.size(); ++j) {
            const auto& c2 = centered2[j];
            if (c1.centered.size() != c2.centered.size() || c1.centered.empty()) continue;

            float corr = -1.0f;
            if (c1.invNorm > 0.0f && c2.invNorm > 0.0f) {
                float dot = 0.0f;
                for (size_t k = 0; k < c1.centered.size(); ++k) {
                    dot += c1.centered[k] * c2.centered[k];
                }
                corr = dot * c1.invNorm * c2.invNorm;
            }

            if (corr > bestCorr) {
                bestCorr = corr;
                bestIdx = static_cast<int>(j);
            }
        }

        if (bestIdx >= 0) {
            matches.emplace_back(static_cast<int>(i), bestIdx, bestCorr);
        }
    }

    return matches;
}
