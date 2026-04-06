#pragma once
#include <vector>

namespace MathUtils {

    // A simple 2D structure capable of holding float pixel values
    struct Matrix2D {
        int width;
        int height;
        std::vector<float> data;

        Matrix2D() : width(0), height(0) {}
        Matrix2D(int w, int h) : width(w), height(h), data(w * h, 0.0f) {}

        inline float& at(int x, int y) {
            return data[y * width + x];
        }

        inline const float& at(int x, int y) const {
            return data[y * width + x];
        }
    };

    // Edge padding mode for convolution
    enum class PaddingMode {
        ZERO,
        REPLICATE
    };

    // Fundamental Operations
    Matrix2D Convolve(const Matrix2D& input, const Matrix2D& kernel, PaddingMode padMode = PaddingMode::REPLICATE);

    // Filter Kernels
    Matrix2D GetGaussianKernel(float sigma);
    Matrix2D GetSobelX();
    Matrix2D GetSobelY();

} // namespace MathUtils
