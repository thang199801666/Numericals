#pragma once

#ifndef NUMERICALS_ITERATIVESOLVERS_H
#define NUMERICALS_ITERATIVESOLVERS_H

#include "SparseMatrix.h"
#include "Vector.h"
#include <cmath>
#include <limits>
#include <functional>
#include <iostream>
#include <vector>
#include <cstring>

namespace CNum
{

    // Conjugate Gradient for symmetric positive-definite sparse matrix A (CSR).
    // Returns solution vector x (promoted floating type). If x0 provided, used as initial guess.
    // Optional progressCallback(iteration, residNorm) will be invoked each iteration if non-null.
    template <typename T, typename U>
    Vector<typename std::common_type<T, U, double>::type>
        ConjugateGradient(const SparseMatrix<T>& A,
            const Vector<U>& b,
            Vector<typename std::common_type<T, U, double>::type> x0 = Vector<typename std::common_type<T, U, double>::type>(),
            double tol = 1e-10,
            std::size_t maxIters = 1000,
            std::function<void(std::size_t, typename std::common_type<T, U, double>::type)> progressCallback = nullptr,
            bool verbose = false)
    {
        using R = typename std::common_type<T, U, double>::type;

        std::size_t n = A.Rows();
        if (b.size() != n) throw std::runtime_error("ConjugateGradient: size mismatch");

        Vector<R> x(n);
        if (x0.size() == n) x = x0;
        else x = Vector<R>(n, R());

        Vector<R> r(n), p(n), Ap(n);
        // r = b - A*x
        A.MatVec(x, Ap);
        for (std::size_t i = 0; i < n; ++i) r[i] = static_cast<R>(b[i]) - Ap[i];
        p = r;

        auto Dot = [](const Vector<R>& a, const Vector<R>& bvec) -> R {
            R s = R();
            std::size_t m = a.size();
            for (std::size_t i = 0; i < m; ++i) s += a[i] * bvec[i];
            return s;
            };

        R rsold = Dot(r, r);
        R resid0 = std::sqrt(static_cast<long double>(rsold));
        if (verbose)
        {
            std::cout << "[CG] n = " << n << ", tol = " << tol << ", maxIters = " << maxIters << "\n";
            std::cout << "[CG] iter = 0, resid = " << resid0 << "\n";
        }
        if (progressCallback) progressCallback(0, resid0);

        if (resid0 <= static_cast<R>(tol)) return x;

        for (std::size_t iter = 1; iter <= maxIters; ++iter)
        {
            A.MatVec(p, Ap);
            R alpha_den = Dot(p, Ap);
            if (std::abs(alpha_den) <= std::numeric_limits<R>::epsilon())
                throw std::runtime_error("ConjugateGradient: breakdown (p'Ap == 0)");
            R alpha = rsold / alpha_den;

            for (std::size_t i = 0; i < n; ++i) x[i] += alpha * p[i];
            for (std::size_t i = 0; i < n; ++i) r[i] -= alpha * Ap[i];

            R rsnew = Dot(r, r);
            R resid = std::sqrt(static_cast<long double>(rsnew));

            if (verbose)
            {
                std::cout << "[CG] iter = " << iter
                    << ", alpha = " << alpha
                    << ", resid = " << resid
                    << ", rsold = " << rsold
                    << ", rsnew = " << rsnew << "\n";
            }

            if (progressCallback) progressCallback(iter, resid);

            if (resid <= static_cast<R>(tol)) break;

            R beta = rsnew / rsold;
            for (std::size_t i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];

            rsold = rsnew;
        }

        return x;
    }

    // High-performance PCG operating entirely on raw double* arrays.
    // Supports both Jacobi (diagInv != null, Lval empty) and IC(0) (Lval non-empty) preconditioning.
    // All vector ops use raw pointers for maximum SIMD opportunity.
    inline Vector<double> PCG_RawPointer(
        const SparseMatrix<double>& A,
        const double* __restrict bptr,
        std::size_t n,
        const double* __restrict diagInv,           // Jacobi: 1/diag(A), or null if IC0
        const std::vector<double>* ic0Lval,         // IC(0) factor values, or null if Jacobi
        double tol,
        std::size_t maxIters,
        std::function<void(std::size_t, double)> progressCallback)
    {
        // Allocate all working vectors as contiguous raw arrays
        std::vector<double> xv(n, 0.0);
        std::vector<double> rv(n);
        std::vector<double> zv(n);
        std::vector<double> pv(n);
        std::vector<double> Apv(n);

        double* __restrict x = xv.data();
        double* __restrict r = rv.data();
        double* __restrict z = zv.data();
        double* __restrict p = pv.data();
        double* __restrict ap = Apv.data();

        // r = b
        std::memcpy(r, bptr, n * sizeof(double));

        // z = M^{-1} r
        if (ic0Lval && !ic0Lval->empty())
        {
            A.IC0Solve(*ic0Lval, r, z);
        }
        else if (diagInv)
        {
            for (std::size_t i = 0; i < n; ++i) z[i] = diagInv[i] * r[i];
        }
        else
        {
            std::memcpy(z, r, n * sizeof(double));
        }

        // p = z
        std::memcpy(p, z, n * sizeof(double));

        double rz_old = 0.0;
        for (std::size_t i = 0; i < n; ++i) rz_old += r[i] * z[i];

        double rr0 = 0.0;
        for (std::size_t i = 0; i < n; ++i) rr0 += r[i] * r[i];
        double resid0 = std::sqrt(rr0);
        if (progressCallback) progressCallback(0, resid0);
        if (resid0 <= tol)
        {
            Vector<double> result(n, 0.0);
            return result;
        }

        for (std::size_t iter = 1; iter <= maxIters; ++iter)
        {
            // Ap = A * p (raw pointer SpMV)
            A.MatVecRaw(p, ap);

            double pAp = 0.0;
            for (std::size_t i = 0; i < n; ++i) pAp += p[i] * ap[i];

            // Relative breakdown check
            double pn2 = 0.0, apn2 = 0.0;
            for (std::size_t i = 0; i < n; ++i) { pn2 += p[i] * p[i]; apn2 += ap[i] * ap[i]; }
            double bkTol = 1e-14 * std::sqrt(pn2 * apn2);
            if (std::abs(pAp) <= bkTol)
                break;

            double alpha = rz_old / pAp;

            // x += alpha * p; r -= alpha * Ap (fused loop for cache locality)
            for (std::size_t i = 0; i < n; ++i)
            {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }

            double rr = 0.0;
            for (std::size_t i = 0; i < n; ++i) rr += r[i] * r[i];
            double resid = std::sqrt(rr);
            if (progressCallback) progressCallback(iter, resid);
            if (resid <= tol) break;

            // z = M^{-1} r
            if (ic0Lval && !ic0Lval->empty())
            {
                A.IC0Solve(*ic0Lval, r, z);
            }
            else if (diagInv)
            {
                for (std::size_t i = 0; i < n; ++i) z[i] = diagInv[i] * r[i];
            }
            else
            {
                std::memcpy(z, r, n * sizeof(double));
            }

            double rz_new = 0.0;
            for (std::size_t i = 0; i < n; ++i) rz_new += r[i] * z[i];

            if (std::abs(rz_old) <= 1e-30) break;

            double beta = rz_new / rz_old;
            for (std::size_t i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
            rz_old = rz_new;
        }

        // Copy result into Vector<double>
        Vector<double> result(n, 0.0);
        std::memcpy(result.data(), x, n * sizeof(double));
        return result;
    }

} // namespace CNum

#endif // NUMERICALS_ITERATIVESOLVERS_H