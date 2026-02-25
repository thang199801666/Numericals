#pragma once

#include "export.h"
#include "Vector.h"
#include <functional>
#include <vector>
#include <string>
#include <array>

namespace CNum
{
    /**
     * @brief Represents a 2D Gauss quadrature point on the reference square [-1,1]x[-1,1].
     * Kept for compatibility with existing code that used QuadPoint2D/GaussQuadrature.
     */
    struct QuadPoint2D
    {
        double xi{ 0.0 };
        double eta{ 0.0 };
        double weight{ 0.0 };
    };

    /**
     * @brief General quadrature helper that mirrors the MATLAB quadrature( quadorder, qt, sdim ) behaviour.
     *
     * Public API:
     * - Quadrature::Compute(quadorder, qt, sdim, W, Q)
     *
     * Where:
     * - quadorder: quadrature order (integer)
     * - qt: quadrature type string, e.g. "GAUSS" or "TRIANGULAR"
     * - sdim: spatial dimension (1,2 or 3)
     * - W: output vector of weights (size = number of quadrature points)
     * - Q: output vector of quadrature points; each entry is a std::vector<double> of length sdim
     */
    class NUMERICALS_API Quadrature
    {
    public:
        // Primary API using std::vector
        static void Compute(std::size_t quadorder,
            const std::string& qt,
            std::size_t sdim,
            std::vector<double>& W,
            std::vector<std::vector<double>>& Q);
    };

    /**
     * @brief Backwards-compatible GaussQuadrature helpers for reference square use.
     *
     * These call into Quadrature::Compute with qt="GAUSS" and sdim=2 and convert results
     * into QuadPoint2D entries.
     */
    class NUMERICALS_API GaussQuadrature
    {
    public:
        static std::vector<QuadPoint2D> GetGaussPoints2D(std::size_t order) noexcept;
        static double Integrate2D(const std::function<double(double, double)>& f, std::size_t order) noexcept;

        /**
         * @brief Compute bilinear Q4 shape functions and their reference derivatives at
         *        Gauss quadrature points for the reference square [-1,1]^2.
         *
         * Populates output vectors (in the same order as Quadrature::Compute's Q for GAUSS, sdim=2):
         * - N:      vector of length npoints, each entry is array<double,4> containing N1..N4 at that point.
         * - dN_dxi: vector of length npoints, each entry is array<double,4> containing dNi/dxi.
         * - dN_deta: vector of length npoints, each entry is array<double,4> containing dNi/deta.
         *
         * The node ordering assumed is:
         *  3 --- 2
         *  |     |
         *  0 --- 1
         *
         * where indices 0..3 correspond to bottom-left, bottom-right, top-right, top-left.
         *
         * The function never throws; on failure it leaves outputs empty.
         */
        static void GetQ4ShapeFunctions(std::size_t order,
            std::vector<std::array<double, 4>>& N,
            std::vector<std::array<double, 4>>& dN_dxi,
            std::vector<std::array<double, 4>>& dN_deta) noexcept;
    };
} // namespace CNum