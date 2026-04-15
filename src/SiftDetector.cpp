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

    auto sampleLinear = [&](const Matrix2D& m, float x, float y) {
        int x0 = (int)std::floor(x);
        int y0 = (int)std::floor(y);
        if (x0 < 0 || x0 >= m.width - 1 || y0 < 0 || y0 >= m.height - 1) return 0.0f;
        float dx = x - x0;
        float dy = y - y0;
        return m.at(x0, y0) * (1.0f - dx) * (1.0f - dy) +
               m.at(x0 + 1, y0) * dx * (1.0f - dy) +
               m.at(x0, y0 + 1) * (1.0f - dx) * dy +
               m.at(x0 + 1, y0 + 1) * dx * dy;
    };
    
    for (const auto& kp : keypoints) {
        int cx = kp.x;
        int cy = kp.y;
        
        // --- 1. ORIENTATION ASSIGNMENT ---
        // Sample orientation in a neighborhood around the keypoint
        std::vector<float> hist(36, 0.0f);
        int radius = 8;
        float sigma = 1.5f * 1.0f; // Simplified scale=1.0
        
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = cx + dx;
                int ny = cy + dy;
                if (nx < 1 || nx >= gaussianImg.width - 1 || ny < 1 || ny >= gaussianImg.height - 1) continue;
                
                float weight = std::exp(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
                float magnitude = mag.at(nx, ny);
                float angle = ori.at(nx, ny);
                if (angle < 0) angle += 2.0f * M_PI;
                
                int bin = (int)(angle / (2.0f * M_PI) * 36.0f) % 36;
                hist[bin] += weight * magnitude;
            }
        }
        
        // Find peak orientation
        float maxVal = 0;
        int maxBin = 0;
        for (int i = 0; i < 36; ++i) {
            if (hist[i] > maxVal) {
                maxVal = hist[i];
                maxBin = i;
            }
        }
        float dominantOri = (maxBin + 0.5f) * (2.0f * M_PI / 36.0f);
        
        // --- 2. DESCRIPTOR EXTRACTION (ROTATED) ---
        // Ensure 16x16 window fits
        if (cx < 10 || cx >= gaussianImg.width - 10 || cy < 10 || cy >= gaussianImg.height - 10) {
            continue;
        }
        
        SiftDescriptor desc;
        desc.x = cx;
        desc.y = cy;
        desc.orientation = dominantOri;
        desc.descriptor.resize(128, 0.0f);
        
        float cosOri = std::cos(dominantOri);
        float sinOri = std::sin(dominantOri);
        
        // Basic descriptor: 4x4 subregions, 8 orientation bins
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                // Bin center in descriptor space (relative to cx, cy)
                // Grid coordinates range from -8 to 8
                std::vector<float> bins(8, 0.0f);
                
                for (int subY = 0; subY < 4; ++subY) {
                    for (int subX = 0; subX < 4; ++subX) {
                        float relX = (c * 4 + subX) - 8.0f + 0.5f;
                        float relY = (r * 4 + subY) - 8.0f + 0.5f;
                        
                        // Rotate coordinates
                        float rotX = cx + (relX * cosOri - relY * sinOri);
                        float rotY = cy + (relX * sinOri + relY * cosOri);
                        
                        float magnitude = sampleLinear(mag, rotX, rotY);
                        float angle = sampleLinear(ori, rotX, rotY);
                        
                        // Adjust angle relative to dominant orientation
                        angle -= dominantOri;
                        while (angle < 0) angle += 2.0f * M_PI;
                        while (angle >= 2.0f * M_PI) angle -= 2.0f * M_PI;
                        
                        int bin = (int)(angle / (2.0f * M_PI) * 8.0f) % 8;
                        
                        // Gaussian weighting for the descriptor window (8.0 sigma)
                        float gWeight = std::exp(-(relX*relX + relY*relY) / (2.0f * 8.0f * 8.0f));
                        bins[bin] += magnitude * gWeight;
                    }
                }
                
                int blockIdx = (r * 4 + c) * 8;
                for (int i = 0; i < 8; ++i) {
                    desc.descriptor[blockIdx + i] = bins[i];
                }
            }
        }
        
        // Normalize the descriptor
        float normSq = 0.0f;
        for (float v : desc.descriptor) normSq += v * v;
        float norm = std::sqrt(normSq);
        if (norm > 1e-6f) {
            for (float& v : desc.descriptor) {
                v /= norm;
                if (v > 0.2f) v = 0.2f;
            }
            normSq = 0.0f;
            for (float v : desc.descriptor) normSq += v * v;
            norm = std::sqrt(normSq);
            if (norm > 1e-6f) {
                for (float& v : desc.descriptor) v /= norm;
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

    const float RATIO_THRESH_SQ = 0.36f; // 0.6 * 0.6
    const size_t MAX_MATCHES = 15;

    for (size_t i = 0; i < set1.size(); ++i) {
        const auto& d1 = set1[i].descriptor;
        float bestDistSq = std::numeric_limits<float>::max();
        float secondBestDistSq = std::numeric_limits<float>::max();
        int bestIdx = -1;

        for (size_t j = 0; j < set2.size(); ++j) {
            const auto& d2 = set2[j].descriptor;
            if (d1.size() != d2.size() || d1.empty()) continue;

            float dot = 0.0f;
            for (size_t k = 0; k < d1.size(); ++k) {
                dot += d1[k] * d2[k];
            }
            float ssd = 2.0f - 2.0f * dot;
            if (ssd < 0) ssd = 0;

            if (ssd < bestDistSq) {
                secondBestDistSq = bestDistSq;
                bestDistSq = ssd;
                bestIdx = static_cast<int>(j);
            } else if (ssd < secondBestDistSq) {
                secondBestDistSq = ssd;
            }
        }

        // Apply Lowe's Ratio Test
        if (bestIdx >= 0 && bestDistSq < RATIO_THRESH_SQ * secondBestDistSq) {
            matches.emplace_back(static_cast<int>(i), bestIdx, bestDistSq);
        }
    }

    // Sort by confidence (best distance first) and limit count
    std::sort(matches.begin(), matches.end(), [](const MatchPair& a, const MatchPair& b) {
        return a.score < b.score;
    });
    
    if (matches.size() > MAX_MATCHES) {
        matches.resize(MAX_MATCHES);
    }

    return matches;
}

std::vector<MatchPair> SiftDetector::MatchDescriptorsNCC(const std::vector<SiftDescriptor>& set1, const std::vector<SiftDescriptor>& set2) {
    std::vector<MatchPair> matches;
    if (set1.empty() || set2.empty()) {
        return matches;
    }

    const float RATIO_THRESH_SQ = 0.64f; // 0.8 * 0.8
    const size_t MAX_MATCHES = 15;

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

    for (size_t i = 0; i < set1.size(); ++i) {
        const auto& c1 = centered1[i];
        float bestCorr = -1.0f;
        float secondBestCorr = -1.0f;
        int bestIdx = -1;

        if (c1.centered.empty()) continue;

        for (size_t j = 0; j < set2.size(); ++j) {
            const auto& c2 = centered2[j];
            if (c1.centered.size() != c2.centered.size()) continue;

            float corr = -1.0f;
            if (c1.invNorm > 0.0f && c2.invNorm > 0.0f) {
                float dot = 0.0f;
                for (size_t k = 0; k < c1.centered.size(); ++k) {
                    dot += c1.centered[k] * c2.centered[k];
                }
                corr = dot * c1.invNorm * c2.invNorm;
            }

            if (corr > bestCorr) {
                secondBestCorr = bestCorr;
                bestCorr = corr;
                bestIdx = static_cast<int>(j);
            } else if (corr > secondBestCorr) {
                secondBestCorr = corr;
            }
        }

        // Ratio test for NCC: Convert to "distance" metric for standard test
        // DistSq = 2 - 2*corr
        float dBestSq = 2.0f - 2.0f * bestCorr;
        float dSecondBestSq = 2.0f - 2.0f * secondBestCorr;
        
        if (bestIdx >= 0 && dBestSq < RATIO_THRESH_SQ * dSecondBestSq) {
            matches.emplace_back(static_cast<int>(i), bestIdx, bestCorr);
        }
    }

    // Sort by confidence (best correlation first) and limit count
    std::sort(matches.begin(), matches.end(), [](const MatchPair& a, const MatchPair& b) {
        return a.score > b.score;
    });

    if (matches.size() > MAX_MATCHES) {
        matches.resize(MAX_MATCHES);
    }

    return matches;
}
