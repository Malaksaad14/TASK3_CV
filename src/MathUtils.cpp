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
        int radius = (int)std::ceil(3.0f * sigma);
        int size = 2 * radius + 1;
        Matrix2D kernel(size, size);
        
        float sum = 0.0f;
        float twoSigmaSquare = 2.0f * sigma * sigma;
        
        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                float val = std::exp(-(x*x + y*y) / twoSigmaSquare);
                kernel.at(x + radius, y + radius) = val;
                sum += val;
            }
        }
        
        // Normalize
        for (size_t i = 0; i < kernel.data.size(); ++i) {
            kernel.data[i] /= sum;
        }
        
        return kernel;
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
