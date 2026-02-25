#pragma once

#ifndef NUMERICALS_INTERPOLATION_H
#define NUMERICALS_INTERPOLATION_H

#include "Vector.h"
#include "CArray.h"
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <type_traits>

namespace CNum
{

    // Interpolation utilities.
    // Public APIs validate inputs and throw std::runtime_error on invalid arguments.
    // Implementations favor clarity and correctness; templates accept arithmetic-like value types.

    // Linear interpolation in 1D (piecewise linear).
    // If xi is outside [x.front(), x.back()] the function linearly extrapolates using the first/last segment.
    template <typename T>
    T linearInterpolation(const Vector<T>& x, const Vector<T>& y, T xi)
    {
        if (x.size() != y.size()) throw std::runtime_error("linearInterpolation: x and y size mismatch");
        if (x.size() < 2) throw std::runtime_error("linearInterpolation: need at least two points");

        // find first element >= xi
        auto begin = x.begin();
        auto end = x.end();
        auto it = std::lower_bound(begin, end, xi);

        std::size_t idx;
        if (it == begin)
        {
            idx = 0; // xi <= x[0] -> extrapolate on [0,1]
        }
        else if (it == end)
        {
            idx = x.size() - 2; // xi > x.back() -> extrapolate on [n-2, n-1]
        }
        else
        {
            idx = static_cast<std::size_t>(std::distance(begin, it)) - 1;
        }

        T x0 = x[idx];
        T x1 = x[idx + 1];
        T y0 = y[idx];
        T y1 = y[idx + 1];

        long double t;
        if (static_cast<long double>(x1) == static_cast<long double>(x0))
            t = 0.0L;
        else
            t = (static_cast<long double>(xi) - static_cast<long double>(x0)) / (static_cast<long double>(x1) - static_cast<long double>(x0));

        long double yi = static_cast<long double>(y0) + t * (static_cast<long double>(y1) - static_cast<long double>(y0));
        return static_cast<T>(yi);
    }

    // Bilinear interpolation on a rectangular grid.
    // `z` is expected to be a 2-D CArray with shape {ny, nx} where nx == x.size(), ny == y.size().
    // Interpolates / extrapolates on the rectangle spanned by x and y.
    template <typename T>
    T bilinearInterpolation(const Vector<T>& x, const Vector<T>& y, const CArray<T>& z, T xi, T yi)
    {
        const std::size_t nx = x.size();
        const std::size_t ny = y.size();
        if (nx < 2 || ny < 2) throw std::runtime_error("bilinearInterpolation: need at least 2 x and 2 y points");
        // z shape must be (ny, nx)
        auto shape = z.shape();
        if (shape.size() != 2 || static_cast<std::size_t>(shape[1]) != nx || static_cast<std::size_t>(shape[0]) != ny)
            throw std::runtime_error("bilinearInterpolation: z shape does not match x/y sizes");

        // find x-interval
        auto itx = std::lower_bound(x.begin(), x.end(), xi);
        std::size_t ix;
        if (itx == x.begin()) ix = 0;
        else if (itx == x.end()) ix = nx - 2;
        else ix = static_cast<std::size_t>(std::distance(x.begin(), itx)) - 1;

        // find y-interval
        auto ity = std::lower_bound(y.begin(), y.end(), yi);
        std::size_t iy;
        if (ity == y.begin()) iy = 0;
        else if (ity == y.end()) iy = ny - 2;
        else iy = static_cast<std::size_t>(std::distance(y.begin(), ity)) - 1;

        T x0 = x[ix];
        T x1 = x[ix + 1];
        T y0 = y[iy];
        T y1 = y[iy + 1];

        long double tx = (static_cast<long double>(x1) == static_cast<long double>(x0)) ? 0.0L :
            (static_cast<long double>(xi) - static_cast<long double>(x0)) / (static_cast<long double>(x1) - static_cast<long double>(x0));
        long double ty = (static_cast<long double>(y1) == static_cast<long double>(y0)) ? 0.0L :
            (static_cast<long double>(yi) - static_cast<long double>(y0)) / (static_cast<long double>(y1) - static_cast<long double>(y0));

        // z indexing: z[iy, ix] etc.
        long double z00 = static_cast<long double>(z.at(shape_t{ iy, ix }));
        long double z10 = static_cast<long double>(z.at(shape_t{ iy, ix + 1 }));
        long double z01 = static_cast<long double>(z.at(shape_t{ iy + 1, ix }));
        long double z11 = static_cast<long double>(z.at(shape_t{ iy + 1, ix + 1 }));

        // bilinear interpolation formula
        long double zxy = (1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z10 + (1 - tx) * ty * z01 + tx * ty * z11;
        return static_cast<T>(zxy);
    }

    // Lagrange polynomial interpolation (direct O(n^2) evaluation).
    // Evaluates the unique polynomial of degree <= n-1 passing through points (x[i], y[i]) at xi.
    template <typename T>
    T lagrangeInterpolation(const Vector<T>& x, const Vector<T>& y, T xi)
    {
        if (x.size() != y.size()) throw std::runtime_error("lagrangeInterpolation: x and y size mismatch");
        const std::size_t n = x.size();
        if (n == 0) throw std::runtime_error("lagrangeInterpolation: empty input");

        // direct evaluation
        long double result = 0.0L;
        for (std::size_t i = 0; i < n; ++i)
        {
            // if xi exactly equals x[i], return y[i] (avoid numerical trouble)
            if (static_cast<long double>(xi) == static_cast<long double>(x[i])) return y[i];

            long double term = static_cast<long double>(y[i]);
            for (std::size_t j = 0; j < n; ++j)
            {
                if (j == i) continue;
                long double denom = static_cast<long double>(x[i]) - static_cast<long double>(x[j]);
                if (denom == 0.0L) throw std::runtime_error("lagrangeInterpolation: duplicate x values");
                term *= (static_cast<long double>(xi) - static_cast<long double>(x[j])) / denom;
            }
            result += term;
        }
        return static_cast<T>(result);
    }

    // Barycentric interpolation (stable O(n) per evaluation after O(n^2) weight computation).
    // Computes barycentric weights and evaluates polynomial. For simplicity this computes weights on each call (O(n^2)).
    template <typename T>
    T barycentricInterpolation(const Vector<T>& x, const Vector<T>& y, T xi)
    {
        if (x.size() != y.size()) throw std::runtime_error("barycentricInterpolation: x and y size mismatch");
        const std::size_t n = x.size();
        if (n == 0) throw std::runtime_error("barycentricInterpolation: empty input");

        std::vector<long double> w(n, 1.0L);

        for (std::size_t j = 0; j < n; ++j)
        {
            for (std::size_t k = 0; k < n; ++k)
            {
                if (j == k) continue;
                long double diff = static_cast<long double>(x[j]) - static_cast<long double>(x[k]);
                if (diff == 0.0L) throw std::runtime_error("barycentricInterpolation: duplicate x values");
                w[j] /= diff;
            }
        }

        long double num = 0.0L;
        long double den = 0.0L;
        for (std::size_t j = 0; j < n; ++j)
        {
            if (static_cast<long double>(xi) == static_cast<long double>(x[j])) return y[j];
            long double term = w[j] / (static_cast<long double>(xi) - static_cast<long double>(x[j]));
            num += term * static_cast<long double>(y[j]);
            den += term;
        }

        return static_cast<T>(num / den);
    }

    // Newton's divided differences interpolation.
    // Builds the divided difference table and evaluates at xi using Newton basis (O(n^2) build, O(n) evaluation).
    template <typename T>
    T newtonsDividedDifferencesInterpolation(const Vector<T>& x, const Vector<T>& y, T xi)
    {
        if (x.size() != y.size()) throw std::runtime_error("newtonsDividedDifferencesInterpolation: x and y size mismatch");
        const std::size_t n = x.size();
        if (n == 0) throw std::runtime_error("newtonsDividedDifferencesInterpolation: empty input");

        // Copy y into a working vector of divided differences (in-place triangular table).
        std::vector<long double> dd(n);
        for (std::size_t i = 0; i < n; ++i) dd[i] = static_cast<long double>(y[i]);

        // build table
        for (std::size_t level = 1; level < n; ++level)
        {
            for (std::size_t i = n - 1; i >= level; --i)
            {
                long double denom = static_cast<long double>(x[i]) - static_cast<long double>(x[i - level]);
                if (denom == 0.0L) throw std::runtime_error("newtonsDividedDifferencesInterpolation: duplicate x values");
                dd[i] = (dd[i] - dd[i - 1]) / denom;
            }
        }

        // evaluate using nested multiplication (Horner-like)
        long double result = dd[n - 1];
        for (std::size_t i = n - 1; i-- > 0; )
        {
            result = result * (static_cast<long double>(xi) - static_cast<long double>(x[i])) + dd[i];
        }

        return static_cast<T>(result);
    }

    // PolynomialInterpolation: convenience wrapper that uses Newton's method by default.
    template <typename T>
    T polynomialInterpolation(const Vector<T>& x, const Vector<T>& y, T xi)
    {
        return newtonsDividedDifferencesInterpolation(x, y, xi);
    }

    // Cubic spline helpers: compute second derivatives (M) for natural or clamped boundary conditions.
    enum class SplineBoundaryType
    {
        Natural,
        Clamped
    };

    // compute second derivatives (m) for cubic spline
    template <typename T>
    std::vector<long double> computeSplineSecondDerivatives(const Vector<T>& xs, const Vector<T>& ys,
        SplineBoundaryType btype, long double fp0 = 0.0L, long double fpn = 0.0L)
    {
        const std::size_t n = xs.size();
        if (n != ys.size()) throw std::runtime_error("computeSplineSecondDerivatives: xs and ys size mismatch");
        if (n < 2) throw std::runtime_error("computeSplineSecondDerivatives: need at least two points");

        std::vector<long double> h(n - 1);
        for (std::size_t i = 0; i + 1 < n; ++i)
        {
            h[i] = static_cast<long double>(xs[i + 1]) - static_cast<long double>(xs[i]);
            if (h[i] == 0.0L) throw std::runtime_error("computeSplineSecondDerivatives: duplicate xs values");
        }

        // tridiagonal system: A * m = rhs, where m are second derivatives
        std::vector<long double> a(n, 0.0L), b(n, 0.0L), c(n, 0.0L), rhs(n, 0.0L);

        if (btype == SplineBoundaryType::Natural)
        {
            // natural spline: second derivatives at endpoints = 0
            b[0] = 1.0L;
            rhs[0] = 0.0L;
            for (std::size_t i = 1; i + 1 < n; ++i)
            {
                a[i] = h[i - 1];
                b[i] = 2.0L * (h[i - 1] + h[i]);
                c[i] = h[i];
                rhs[i] = 6.0L * ((static_cast<long double>(ys[i + 1]) - static_cast<long double>(ys[i])) / h[i]
                    - (static_cast<long double>(ys[i]) - static_cast<long double>(ys[i - 1])) / h[i - 1]);
            }
            a[n - 1] = 0.0L;
            b[n - 1] = 1.0L;
            rhs[n - 1] = 0.0L;
        }
        else // Clamped
        {
            // clamped: first derivatives at endpoints provided (fp0, fpn)
            b[0] = 2.0L * h[0];
            c[0] = h[0];
            rhs[0] = 6.0L * ((static_cast<long double>(ys[1]) - static_cast<long double>(ys[0])) / h[0] - fp0);

            for (std::size_t i = 1; i + 1 < n; ++i)
            {
                a[i] = h[i - 1];
                b[i] = 2.0L * (h[i - 1] + h[i]);
                c[i] = h[i];
                rhs[i] = 6.0L * ((static_cast<long double>(ys[i + 1]) - static_cast<long double>(ys[i])) / h[i]
                    - (static_cast<long double>(ys[i]) - static_cast<long double>(ys[i - 1])) / h[i - 1]);
            }

            a[n - 1] = h[n - 2];
            b[n - 1] = 2.0L * h[n - 2];
            rhs[n - 1] = 6.0L * (fpn - (static_cast<long double>(ys[n - 1]) - static_cast<long double>(ys[n - 2])) / h[n - 2]);
        }

        // Solve tridiagonal system using Thomas algorithm
        // modify coefficients
        std::vector<long double> cp(n, 0.0L), dp(n, 0.0L);
        cp[0] = (b[0] == 0.0L) ? 0.0L : (c[0] / b[0]);
        dp[0] = (b[0] == 0.0L) ? 0.0L : (rhs[0] / b[0]);
        for (std::size_t i = 1; i < n; ++i)
        {
            long double denom = b[i] - a[i] * cp[i - 1];
            if (denom == 0.0L) throw std::runtime_error("computeSplineSecondDerivatives: singular system");
            cp[i] = (i + 1 < n) ? (c[i] / denom) : 0.0L;
            dp[i] = (rhs[i] - a[i] * dp[i - 1]) / denom;
        }

        std::vector<long double> m(n, 0.0L);
        m[n - 1] = dp[n - 1];
        for (std::size_t i = n - 1; i-- > 0; )
        {
            m[i] = dp[i] - cp[i] * m[i + 1];
        }

        return m;
    }

    // Evaluate cubic spline at xi using precomputed second derivatives m.
    template <typename T>
    T evaluateCubicSpline(const Vector<T>& xs, const Vector<T>& ys, const std::vector<long double>& m, T xi)
    {
        const std::size_t n = xs.size();
        if (n != ys.size()) throw std::runtime_error("evaluateCubicSpline: xs and ys size mismatch");
        if (m.size() != n) throw std::runtime_error("evaluateCubicSpline: m size mismatch");
        if (n < 2) throw std::runtime_error("evaluateCubicSpline: need at least two points");

        // find interval
        auto it = std::lower_bound(xs.begin(), xs.end(), xi);
        std::size_t idx;
        if (it == xs.begin()) idx = 0;
        else if (it == xs.end()) idx = n - 2;
        else idx = static_cast<std::size_t>(std::distance(xs.begin(), it)) - 1;

        long double x0 = static_cast<long double>(xs[idx]);
        long double x1 = static_cast<long double>(xs[idx + 1]);
        long double y0 = static_cast<long double>(ys[idx]);
        long double y1 = static_cast<long double>(ys[idx + 1]);
        long double m0 = m[idx];
        long double m1 = m[idx + 1];
        long double h = x1 - x0;
        if (h == 0.0L) throw std::runtime_error("evaluateCubicSpline: zero interval width");

        long double t = (static_cast<long double>(xi) - x0) / h;

        // cubic spline Hermite-like form using second derivatives:
        long double a = (m1 - m0) / (6.0L * h);
        long double b = m0 / 2.0L;
        long double c = (static_cast<long double>(y1) - static_cast<long double>(y0)) / h - (2.0L * h * m0 + h * m1) / 6.0L;
        long double dx = static_cast<long double>(xi) - x0;

        long double yi = static_cast<long double>(y0) + static_cast<long double>(c) * dx + static_cast<long double>(b) * dx * dx + static_cast<long double>(a) * dx * dx * dx;
        return static_cast<T>(yi);
    }

    // Natural cubic spline interpolation
    template <typename T>
    T naturalCubicSplineInterpolation(const Vector<T>& x, const Vector<T>& y, T xi)
    {
        auto m = computeSplineSecondDerivatives(x, y, SplineBoundaryType::Natural);
        return evaluateCubicSpline(x, y, m, xi);
    }

    // Clamped cubic spline interpolation: requires endpoint derivatives fp0 (at x[0]) and fpn (at x[n-1])
    template <typename T>
    T clampedCubicSplineInterpolation(const Vector<T>& x, const Vector<T>& y, T xi, T fp0, T fpn)
    {
        auto m = computeSplineSecondDerivatives(x, y, SplineBoundaryType::Clamped, static_cast<long double>(fp0), static_cast<long double>(fpn));
        return evaluateCubicSpline(x, y, m, xi);
    }

    // --- Convenience aliases preserving user's requested PascalCase names ---
    template <typename T>
    inline T LinearInterpolation(const Vector<T>& x, const Vector<T>& y, T xi) { return linearInterpolation(x, y, xi); }

    template <typename T>
    inline T BilinearInterpolation(const Vector<T>& x, const Vector<T>& y, const CArray<T>& z, T xi, T yi) { return bilinearInterpolation(x, y, z, xi, yi); }

    template <typename T>
    inline T PolynomialInterpolation(const Vector<T>& x, const Vector<T>& y, T xi) { return polynomialInterpolation(x, y, xi); }

    template <typename T>
    inline T LagrangeInterpolation(const Vector<T>& x, const Vector<T>& y, T xi) { return lagrangeInterpolation(x, y, xi); }

    template <typename T>
    inline T BarycentricInterpolation(const Vector<T>& x, const Vector<T>& y, T xi) { return barycentricInterpolation(x, y, xi); }

    template <typename T>
    inline T NewtonsDividedDifferencesInterpolation(const Vector<T>& x, const Vector<T>& y, T xi) { return newtonsDividedDifferencesInterpolation(x, y, xi); }

    template <typename T>
    inline T CubicSplineNatural(const Vector<T>& x, const Vector<T>& y, T xi) { return naturalCubicSplineInterpolation(x, y, xi); }

    template <typename T>
    inline T CubicSplineClamped(const Vector<T>& x, const Vector<T>& y, T xi, T fp0, T fpn) { return clampedCubicSplineInterpolation(x, y, xi, fp0, fpn); }

} // namespace CNum

#endif // NUMERICALS_INTERPOLATION_H