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
// ahna bnhsb GX and GY w b3den bnhsb Ix^2 w Iy^2 w Ix*Iy
void HarrisDetector::ComputeGradientsAndProducts(
    const Matrix2D& img,
    Matrix2D& Ix2, Matrix2D& Iy2, Matrix2D& Ixy) {
    
    // Sobel filters are separable:
    // Sobel X = [1, 2, 1]^T * [-1, 0, 1]
    //da kda l kernel bta3 Sobel x ashan nhsb GX
    std::vector<float> kernelX_h = {-1.0f, 0.0f, 1.0f};
    std::vector<float> kernelX_v = {1.0f, 2.0f, 1.0f};
    
    // Sobel Y = [-1, 0, 1]^T * [1, 2, 1]
    //da kda l kernel bta3 Sobel y ashan nhsb GY
    std::vector<float> kernelY_h = {1.0f, 2.0f, 1.0f};
    std::vector<float> kernelY_v = {-1.0f, 0.0f, 1.0f};
    //hna bnhsb Gx w Gy
    Matrix2D Ix = ConvolveSeparable(img, kernelX_h, kernelX_v, PaddingMode::REPLICATE);
    Matrix2D Iy = ConvolveSeparable(img, kernelY_h, kernelY_v, PaddingMode::REPLICATE);
    
    int w = img.width;
    int h = img.height;
    Ix2 = Matrix2D(w, h);
    Iy2 = Matrix2D(w, h);
    Ixy = Matrix2D(w, h);
    
    for (int i = 0; i < w * h; ++i) {
        float ix = Ix.data[i];
        float iy = Iy.data[i];
        Ix2.data[i] = ix * ix;
        Iy2.data[i] = iy * iy;
        Ixy.data[i] = ix * iy;
    }
}

std::vector<KeyPoint> HarrisDetector::DetectHarris(const Matrix2D& img, float k, float threshold, int minPoints, int nmsRadius) {
    Matrix2D Ix2, Iy2, Ixy;
    ComputeGradientsAndProducts(img, Ix2, Iy2, Ixy); // b3d ma hsbna IX2, IY2, IXY
    
    // Apply Gaussian blur (sigma = 1.0) astkhdmna gaussian filter
    std::vector<float> gauss1D = GetGaussianKernel1D(1.0f);
    Matrix2D Sxx = ConvolveSeparable(Ix2, gauss1D, gauss1D); // gaussian filter * IX2 bnfs l tare2a l hsbna beha l GX wl GY
    Matrix2D Syy = ConvolveSeparable(Iy2, gauss1D, gauss1D); // gaussian filter * IY2
    Matrix2D Sxy = ConvolveSeparable(Ixy, gauss1D, gauss1D); // gaussian filter * IXY
    
    Matrix2D responseMap(img.width, img.height);
    // hena bnhsb b2a l det wl trace ashan nhsb l response 
    // ngeb awl element mn kol matrix w nroh nhsb l det wl trace w b3den ngeb tany element 
    // lhad ma nkhlshom
    for (int i = 0; i < img.width * img.height; ++i) {
        float sxx = Sxx.data[i];
        float syy = Syy.data[i];
        float sxy = Sxy.data[i];
        
        float det = sxx * syy - sxy * sxy;
        float trace = sxx + syy;
        // hna bnhsb l harris response (R) l kol pixel
        responseMap.data[i] = det - k * trace * trace; 
    }
    
    return NonMaximumSuppression(responseMap, threshold, nmsRadius);
}

std::vector<KeyPoint> HarrisDetector::DetectLambdaMinus(const Matrix2D& img, float threshold, int minPoints, int nmsRadius) {
    Matrix2D Ix2, Iy2, Ixy;
    ComputeGradientsAndProducts(img, Ix2, Iy2, Ixy);
    
    std::vector<float> gauss1D = GetGaussianKernel1D(1.0f);
    Matrix2D Sxx = ConvolveSeparable(Ix2, gauss1D, gauss1D);
    Matrix2D Syy = ConvolveSeparable(Iy2, gauss1D, gauss1D);
    Matrix2D Sxy = ConvolveSeparable(Ixy, gauss1D, gauss1D);
    
    Matrix2D responseMap(img.width, img.height);
    
    for (int i = 0; i < img.width * img.height; ++i) {
        float sxx = Sxx.data[i];
        float syy = Syy.data[i];
        float sxy = Sxy.data[i];
        
        float trace = sxx + syy;
        float det = sxx * syy - sxy * sxy;
        //We derive λ from det(M − λI)=0 → quadratic equation → gives trace and determinant formula(λ-(trace)λ +det=0)
        // lma bnhl l equation bta3t l eigenvalue bnwsl ll formula de l feha l trace wl det (λ-(trace)λ +det=0) 
        float inner = trace * trace - 4 * det;
        float lambda_min = 0.0f;
        if (inner >= 0.0f) {
            lambda_min = (trace - std::sqrt(inner)) / 2.0f; // we b3den ngeb l min eigen value
        }
        
        responseMap.data[i] = lambda_min;
    }
    
    return NonMaximumSuppression(responseMap, threshold, nmsRadius);
}

std::vector<KeyPoint> HarrisDetector::NonMaximumSuppression(const Matrix2D& responseMap, float threshold, int radius) {
    // hena vector hnkhzn feh l points l f3ln corner
    std::vector<KeyPoint> keypoints;
    // hnmshy 3l image bs hnbd2 mn l centered pixel 
    // w nshof hal hya akbr mn l neighbors ely 7waleha wala laa
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
    
    // hnrtb l points hsb l strength  
    std::sort(keypoints.begin(), keypoints.end(), [](const KeyPoint& a, const KeyPoint& b) {
        return a.response > b.response;
    });
    
    return keypoints;
} 
