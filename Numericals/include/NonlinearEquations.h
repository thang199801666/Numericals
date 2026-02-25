#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace CNum
{
    /**
     * @brief Result returned by iterative root-finding methods.
     */
    struct RootResult
    {
        double root{ std::numeric_limits<double>::quiet_NaN() };
        std::size_t iterations{ 0 };
        bool converged{ false };
    };

    /**
     * @brief Utilities for solving nonlinear equations f(x) = 0.
     *
     * Public function names use PascalCase for the project's naming convention.
     * Functions accept std::function<double(double)> for the target function (and derivative where required).
     */
    class NonlinearEquations
    {
    public:
        using Func = std::function<double(double)>;

        /**
         * @brief Linear incremental scan to locate subintervals where f changes sign.
         *
         * Scans [a, b] in steps of h and returns a vector of interval midpoints where sign changes occur.
         * This routine never throws; invalid inputs yield an empty result.
         *
         * @param f Function f(x).
         * @param a Left endpoint.
         * @param b Right endpoint.
         * @param h Step size (must be > 0).
         * @return Vector of candidate roots (midpoints of sign-change intervals).
         */
        static std::vector<double> LinearIncrementalMethod(const Func& f, double a, double b, double h) noexcept
        {
            std::vector<double> roots;
            if (h <= 0.0 || a > b)
            {
                return roots;
            }

            double x = a;
            double fx = f(x);
            while (x + h <= b)
            {
                double xn = x + h;
                double fxn = f(xn);
                if (std::isnan(fx) || std::isnan(fxn))
                {
                    // skip intervals with NaN
                }
                else if (fx == 0.0)
                {
                    roots.push_back(x);
                }
                else if (fx * fxn < 0.0)
                {
                    roots.push_back((x + xn) * 0.5);
                }
                x = xn;
                fx = fxn;
            }

            // check final endpoint
            if (x <= b)
            {
                double fxend = f(b);
                if (!std::isnan(fxend) && fxend == 0.0)
                {
                    roots.push_back(b);
                }
            }

            // remove duplicates (within machine epsilon)
            std::sort(roots.begin(), roots.end());
            roots.erase(std::unique(roots.begin(), roots.end(), [](double p, double q) {
                return std::abs(p - q) <= std::numeric_limits<double>::epsilon() * std::max(1.0, std::max(std::abs(p), std::abs(q)));
                }), roots.end());

            return roots;
        }

        /**
         * @brief Bisection method for f(x) = 0 on [a, b].
         *
         * Throws std::invalid_argument if f(a) and f(b) do not bracket a root (and neither equals zero).
         *
         * @param f Function f(x).
         * @param a Left bracket.
         * @param b Right bracket.
         * @param tol Tolerance for |b-a| / 2 or |f(m)|.
         * @param maxIter Maximum iterations.
         * @return RootResult containing root, iterations, and convergence flag.
         */
        static RootResult BisectionMethod(const Func& f, double a, double b, double tol = 1e-12, std::size_t maxIter = 100)
        {
            double fa = f(a);
            double fb = f(b);

            if (std::isnan(fa) || std::isnan(fb))
            {
                throw std::invalid_argument("Function returned NaN at interval endpoints.");
            }

            if (fa == 0.0)
            {
                return RootResult{ a, 0, true };
            }
            if (fb == 0.0)
            {
                return RootResult{ b, 0, true };
            }
            if (fa * fb > 0.0)
            {
                throw std::invalid_argument("Bisection method requires a sign change on the interval [a, b].");
            }

            double left = a;
            double right = b;
            RootResult res;
            for (std::size_t iter = 1; iter <= maxIter; ++iter)
            {
                double mid = 0.5 * (left + right);
                double fm = f(mid);

                res.iterations = iter;
                res.root = mid;

                if (std::abs(fm) <= tol || (right - left) * 0.5 <= tol)
                {
                    res.converged = true;
                    return res;
                }

                if (fa * fm < 0.0)
                {
                    right = mid;
                    fb = fm;
                }
                else
                {
                    left = mid;
                    fa = fm;
                }
            }

            res.converged = false;
            return res;
        }

        /**
         * @brief Secant method for f(x) = 0.
         *
         * @param f Function f(x).
         * @param x0 First initial guess.
         * @param x1 Second initial guess.
         * @param tol Tolerance on |x_{n+1} - x_n|.
         * @param maxIter Maximum iterations.
         * @return RootResult with approximate root.
         */
        static RootResult TheSecantMethod(const Func& f, double x0, double x1, double tol = 1e-12, std::size_t maxIter = 100)
        {
            double f0 = f(x0);
            double f1 = f(x1);
            RootResult res;

            for (std::size_t iter = 1; iter <= maxIter; ++iter)
            {
                if (std::abs(f1 - f0) <= std::numeric_limits<double>::epsilon())
                {
                    res.root = x1;
                    res.iterations = iter;
                    res.converged = false;
                    return res;
                }

                double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
                res.iterations = iter;
                res.root = x2;

                if (std::abs(x2 - x1) <= tol)
                {
                    res.converged = true;
                    return res;
                }

                x0 = x1;
                f0 = f1;
                x1 = x2;
                f1 = f(x1);
            }

            res.converged = false;
            return res;
        }

        /**
         * @brief False position (Regula Falsi) method for f(x) = 0 on [a, b].
         *
         * Throws std::invalid_argument if f(a) and f(b) do not bracket a root.
         *
         * @param f Function f(x).
         * @param a Left bracket.
         * @param b Right bracket.
         * @param tol Tolerance for root or function value.
         * @param maxIter Maximum iterations.
         * @return RootResult with approximate root.
         */
        static RootResult FalsePositioningMethod(const Func& f, double a, double b, double tol = 1e-12, std::size_t maxIter = 100)
        {
            double fa = f(a);
            double fb = f(b);

            if (std::isnan(fa) || std::isnan(fb))
            {
                throw std::invalid_argument("Function returned NaN at interval endpoints.");
            }

            if (fa == 0.0)
            {
                return RootResult{ a, 0, true };
            }
            if (fb == 0.0)
            {
                return RootResult{ b, 0, true };
            }
            if (fa * fb > 0.0)
            {
                throw std::invalid_argument("False position requires a sign change on the interval [a, b].");
            }

            double left = a;
            double right = b;
            double fl = fa;
            double fr = fb;
            RootResult res;

            for (std::size_t iter = 1; iter <= maxIter; ++iter)
            {
                double x = (left * fr - right * fl) / (fr - fl); // interpolation
                double fx = f(x);

                res.iterations = iter;
                res.root = x;

                if (std::abs(fx) <= tol)
                {
                    res.converged = true;
                    return res;
                }

                if (fl * fx < 0.0)
                {
                    right = x;
                    fr = fx;
                }
                else
                {
                    left = x;
                    fl = fx;
                }
            }

            res.converged = false;
            return res;
        }

        /**
         * @brief Fixed-point iteration for x = g(x).
         *
         * Converges only when g is a contraction near the fixed point. The caller supplies g.
         *
         * @param g Iteration function g(x).
         * @param x0 Initial guess.
         * @param tol Tolerance on |x_{n+1} - x_n|.
         * @param maxIter Maximum iterations.
         * @return RootResult containing the fixed point (if converged).
         */
        static RootResult FixedPointIteration(const Func& g, double x0, double tol = 1e-12, std::size_t maxIter = 100)
        {
            RootResult res;
            double x = x0;
            for (std::size_t iter = 1; iter <= maxIter; ++iter)
            {
                double xn = g(x);
                res.iterations = iter;
                res.root = xn;

                if (!std::isfinite(xn))
                {
                    res.converged = false;
                    return res;
                }

                if (std::abs(xn - x) <= tol)
                {
                    res.converged = true;
                    return res;
                }

                x = xn;
            }

            res.converged = false;
            return res;
        }

        /**
         * @brief Newton-Raphson method for f(x) = 0 using derivative df.
         *
         * If derivative is zero or produces NaN/Inf the method returns with converged = false.
         *
         * @param f Function f(x).
         * @param df Derivative f'(x).
         * @param x0 Initial guess.
         * @param tol Tolerance on |x_{n+1} - x_n|.
         * @param maxIter Maximum iterations.
         * @return RootResult with approximate root.
         */
        static RootResult NewtonRaphsonMethod(const Func& f, const Func& df, double x0, double tol = 1e-12, std::size_t maxIter = 100)
        {
            RootResult res;
            double x = x0;

            for (std::size_t iter = 1; iter <= maxIter; ++iter)
            {
                double fx = f(x);
                double dfx = df(x);

                if (!std::isfinite(fx) || !std::isfinite(dfx))
                {
                    res.iterations = iter;
                    res.root = x;
                    res.converged = false;
                    return res;
                }

                if (std::abs(dfx) <= std::numeric_limits<double>::epsilon())
                {
                    res.iterations = iter;
                    res.root = x;
                    res.converged = false;
                    return res;
                }

                double xn = x - fx / dfx;
                res.iterations = iter;
                res.root = xn;

                if (std::abs(xn - x) <= tol)
                {
                    res.converged = true;
                    return res;
                }

                x = xn;
            }

            res.converged = false;
            return res;
        }
    };
} // namespace Numericals