#include "MathUtils.h"
#include <cmath>
#include <algorithm>

namespace MathUtils {

    Matrix2D Convolve(const Matrix2D& input, const Matrix2D& kernel, PaddingMode padMode) {
        Matrix2D output(input.width, input.height);
        int kw = kernel.width;
        int kh = kernel.height;
        int kw_radius = kw / 2;
        int kh_radius = kh / 2;

        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                float sum = 0.0f;

                for (int ky = 0; ky < kh; ++ky) {
                    for (int kx = 0; kx < kw; ++kx) {
                        int iy = y + ky - kh_radius;
                        int ix = x + kx - kw_radius;

                        float pixelVal = 0.0f;
                        if (ix >= 0 && ix < input.width && iy >= 0 && iy < input.height) {
                            pixelVal = input.at(ix, iy);
                        } else if (padMode == PaddingMode::REPLICATE) {
                            int cx = std::max(0, std::min(ix, input.width - 1));
                            int cy = std::max(0, std::min(iy, input.height - 1));
                            pixelVal = input.at(cx, cy);
                        }

                        sum += pixelVal * kernel.at(kx, ky);
                    }
                }
                output.at(x, y) = sum;
            }
        }
        return output;
    }

    Matrix2D GetGaussianKernel(float sigma) {
        std::vector<float> kernel1D = GetGaussianKernel1D(sigma);
        int size = kernel1D.size();
        Matrix2D kernel(size, size);
        
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                kernel.at(x, y) = kernel1D[x] * kernel1D[y];
            }
        }
        return kernel;
    }

    std::vector<float> GetGaussianKernel1D(float sigma) {
        int radius = (int)std::ceil(3.0f * sigma);
        int size = 2 * radius + 1;
        std::vector<float> kernel(size);
        
        float sum = 0.0f;
        float twoSigmaSquare = 2.0f * sigma * sigma;
        
        for (int i = 0; i < size; ++i) {
            int x = i - radius;
            float val = std::exp(-(x*x) / twoSigmaSquare);
            kernel[i] = val;
            sum += val;
        }
        
        // Normalize
        for (int i = 0; i < size; ++i) {
            kernel[i] /= sum;
        }
        
        return kernel;
    }

    Matrix2D ConvolveSeparable(const Matrix2D& input, const std::vector<float>& hKernel, const std::vector<float>& vKernel, PaddingMode padMode) {
        int width = input.width;
        int height = input.height;
        Matrix2D temp(width, height);
        Matrix2D output(width, height);

        int kw = hKernel.size();
        int kh = vKernel.size();
        int kw_radius = kw / 2;
        int kh_radius = kh / 2;

        // Horizontal Pass
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (int k = 0; k < kw; ++k) {
                    int ix = x + k - kw_radius;
                    float val;
                    if (ix >= 0 && ix < width) {
                        val = input.at(ix, y);
                    } else if (padMode == PaddingMode::REPLICATE) {
                        val = input.at(std::max(0, std::min(ix, width - 1)), y);
                    } else {
                        val = 0.0f;
                    }
                    sum += val * hKernel[k];
                }
                temp.at(x, y) = sum;
            }
        }

        // Vertical Pass
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (int k = 0; k < kh; ++k) {
                    int iy = y + k - kh_radius;
                    float val;
                    if (iy >= 0 && iy < height) {
                        val = temp.at(x, iy);
                    } else if (padMode == PaddingMode::REPLICATE) {
                        val = temp.at(x, std::max(0, std::min(iy, height - 1)));
                    } else {
                        val = 0.0f;
                    }
                    sum += val * vKernel[k];
                }
                output.at(x, y) = sum;
            }
        }

        return output;
    }

    Matrix2D GetSobelX() {
        Matrix2D kernel(3, 3);
        kernel.at(0, 0) = -1.0f; kernel.at(1, 0) = 0.0f; kernel.at(2, 0) = 1.0f;
        kernel.at(0, 1) = -2.0f; kernel.at(1, 1) = 0.0f; kernel.at(2, 1) = 2.0f;
        kernel.at(0, 2) = -1.0f; kernel.at(1, 2) = 0.0f; kernel.at(2, 2) = 1.0f;
        return kernel;
    }

    Matrix2D GetSobelY() {
        Matrix2D kernel(3, 3);
        kernel.at(0, 0) = -1.0f; kernel.at(1, 0) = -2.0f; kernel.at(2, 0) = -1.0f;
        kernel.at(0, 1) = 0.0f;   kernel.at(1, 1) = 0.0f;  kernel.at(2, 1) = 0.0f;
        kernel.at(0, 2) = 1.0f;  kernel.at(1, 2) = 2.0f;  kernel.at(2, 2) = 1.0f;
        return kernel;
    }

} // namespace MathUtils
