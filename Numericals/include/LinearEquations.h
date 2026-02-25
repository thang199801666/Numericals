#pragma once

#ifndef NUMERICALS_LINEAREQUATIONS_H
#define NUMERICALS_LINEAREQUATIONS_H

#include "Matrix.h"
#include "Vector.h"
#include <vector>
#include <tuple>
#include <utility>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace CNum
{

    // Linear equation utilities and solvers.
    // Methods throw std::runtime_error on invalid input (singular, size mismatch, non-symmetric where required).
    template <typename T>
    class LinearEquations
    {
    public:
        using value_type = T;

        // Gaussian elimination (forward elimination + back substitution) with partial pivoting.
        // Returns solution vector x for A * x = b.
        template <typename U>
        static Vector<typename std::common_type<T, U, double>::type> GaussianElimination(const Matrix<T>& A, const Vector<U>& b)
        {
            using R = typename std::common_type<T, U, double>::type;
            if (!A.is_square()) throw std::runtime_error("GaussianElimination: A must be square");
            std::size_t n = A.rows();
            if (b.size() != n) throw std::runtime_error("GaussianElimination: size mismatch");

            // build augmented matrix
            std::vector<std::vector<R>> a(n, std::vector<R>(n + 1));
            for (std::size_t i = 0; i < n; ++i)
            {
                for (std::size_t j = 0; j < n; ++j) a[i][j] = static_cast<R>(A(i, j));
                a[i][n] = static_cast<R>(b[i]);
            }

            // forward elimination with partial pivoting
            for (std::size_t k = 0; k < n; ++k)
            {
                // pivot selection
                std::size_t pivot = k;
                R maxabs = std::abs(a[k][k]);
                for (std::size_t i = k + 1; i < n; ++i)
                {
                    R val = std::abs(a[i][k]);
                    if (val > maxabs) { maxabs = val; pivot = i; }
                }
                if (std::abs(a[pivot][k]) <= std::numeric_limits<R>::epsilon() * R(1e2))
                    throw std::runtime_error("GaussianElimination: matrix is singular or numerically unstable");

                if (pivot != k) std::swap(a[pivot], a[k]);

                // eliminate below
                for (std::size_t i = k + 1; i < n; ++i)
                {
                    R factor = a[i][k] / a[k][k];
                    for (std::size_t j = k; j <= n; ++j)
                        a[i][j] -= factor * a[k][j];
                }
            }

            // back substitution
            Vector<R> x(n);
            for (int i = static_cast<int>(n) - 1; i >= 0; --i)
            {
                R s = a[i][n];
                for (std::size_t j = i + 1; j < n; ++j) s -= a[i][j] * x[j];
                x[i] = s / a[i][i];
            }
            return x;
        }

        // Gauss-Jordan elimination producing the inverse of A (if invertible).
        // Also provided overload to solve A*x = b via Gauss-Jordan on augmented matrix.
        static Matrix<typename std::common_type<T, double>::type> GaussJordanElimination(const Matrix<T>& A)
        {
            using R = typename std::common_type<T, double>::type;
            if (!A.is_square()) throw std::runtime_error("GaussJordanElimination: A must be square");
            std::size_t n = A.rows();
            if (n == 0) return Matrix<R>(0, 0);

            // augmented [A | I]
            std::vector<std::vector<R>> a(n, std::vector<R>(2 * n));
            for (std::size_t i = 0; i < n; ++i)
            {
                for (std::size_t j = 0; j < n; ++j) a[i][j] = static_cast<R>(A(i, j));
                for (std::size_t j = 0; j < n; ++j) a[i][n + j] = (i == j) ? R(1) : R(0);
            }

            for (std::size_t i = 0; i < n; ++i)
            {
                // pivot selection
                std::size_t pivot = i;
                R maxabs = std::abs(a[i][i]);
                for (std::size_t r = i + 1; r < n; ++r)
                {
                    R val = std::abs(a[r][i]);
                    if (val > maxabs) { maxabs = val; pivot = r; }
                }
                if (std::abs(a[pivot][i]) <= std::numeric_limits<R>::epsilon() * R(1e2))
                    throw std::runtime_error("GaussJordanElimination: matrix is singular or numerically unstable");

                if (pivot != i) std::swap(a[pivot], a[i]);

                // normalize pivot row
                R diag = a[i][i];
                for (std::size_t j = 0; j < 2 * n; ++j) a[i][j] /= diag;

                // eliminate other rows
                for (std::size_t r = 0; r < n; ++r)
                {
                    if (r == i) continue;
                    R factor = a[r][i];
                    for (std::size_t j = 0; j < 2 * n; ++j) a[r][j] -= factor * a[i][j];
                }
            }

            Matrix<R> inv(n, n);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    inv(i, j) = a[i][n + j];
            return inv;
        }

        template <typename U>
        static Vector<typename std::common_type<T, U, double>::type> GaussJordanElimination(const Matrix<T>& A, const Vector<U>& b)
        {
            using R = typename std::common_type<T, U, double>::type;
            if (!A.is_square()) throw std::runtime_error("GaussJordanElimination: A must be square");
            std::size_t n = A.rows();
            if (b.size() != n) throw std::runtime_error("GaussJordanElimination: size mismatch");
            if (n == 0) return Vector<R>(0);

            // augmented [A | b]
            std::vector<std::vector<R>> a(n, std::vector<R>(n + 1));
            for (std::size_t i = 0; i < n; ++i)
            {
                for (std::size_t j = 0; j < n; ++j) a[i][j] = static_cast<R>(A(i, j));
                a[i][n] = static_cast<R>(b[i]);
            }

            for (std::size_t i = 0; i < n; ++i)
            {
                // pivot selection
                std::size_t pivot = i;
                R maxabs = std::abs(a[i][i]);
                for (std::size_t r = i + 1; r < n; ++r)
                {
                    R val = std::abs(a[r][i]);
                    if (val > maxabs) { maxabs = val; pivot = r; }
                }
                if (std::abs(a[pivot][i]) <= std::numeric_limits<R>::epsilon() * R(1e2))
                    throw std::runtime_error("GaussJordanElimination: matrix is singular or numerically unstable");

                if (pivot != i) std::swap(a[pivot], a[i]);

                // normalize pivot row
                R diag = a[i][i];
                for (std::size_t j = i; j <= n; ++j) a[i][j] /= diag;

                // eliminate other rows
                for (std::size_t r = 0; r < n; ++r)
                {
                    if (r == i) continue;
                    R factor = a[r][i];
                    for (std::size_t j = i; j <= n; ++j) a[r][j] -= factor * a[i][j];
                }
            }

            Vector<R> x(n);
            for (std::size_t i = 0; i < n; ++i) x[i] = a[i][n];
            return x;
        }

        // LU decomposition with partial pivoting.
        // Returns tuple (L, U, perm) where perm is a permutation vector such that P*A = L*U.
        // L has unit diagonal.
        static std::tuple< Matrix<typename std::common_type<T, double>::type>,
            Matrix<typename std::common_type<T, double>::type>,
            Vector<std::size_t> >
            LUDecomposition(const Matrix<T>& A)
        {
            using R = typename std::common_type<T, double>::type;
            if (!A.is_square()) throw std::runtime_error("LUDecomposition: A must be square");
            std::size_t n = A.rows();

            // copy A into U (working), initialize L to identity
            Matrix<R> U = Matrix<R>(A.rows(), A.cols());
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    U(i, j) = static_cast<R>(A(i, j));

            Matrix<R> L = Matrix<R>::Eye(n);
            Vector<std::size_t> perm(n);
            for (std::size_t i = 0; i < n; ++i) perm[i] = i;

            for (std::size_t k = 0; k < n; ++k)
            {
                // partial pivot on column k (rows k..n-1)
                std::size_t pivot = k;
                R maxabs = std::abs(U(k, k));
                for (std::size_t i = k + 1; i < n; ++i)
                {
                    R val = std::abs(U(i, k));
                    if (val > maxabs) { maxabs = val; pivot = i; }
                }
                if (std::abs(U(pivot, k)) <= std::numeric_limits<R>::epsilon() * R(1e2))
                    throw std::runtime_error("LUDecomposition: matrix is singular or numerically unstable");

                if (pivot != k)
                {
                    // swap rows in U
                    for (std::size_t j = 0; j < n; ++j) std::swap(U(k, j), U(pivot, j));
                    // swap previous L columns (only columns [0..k-1] are populated)
                    for (std::size_t j = 0; j < k; ++j) std::swap(L(k, j), L(pivot, j));
                    // record permutation
                    std::swap(perm[k], perm[pivot]);
                }

                // compute multipliers and eliminate
                for (std::size_t i = k + 1; i < n; ++i)
                {
                    R mult = U(i, k) / U(k, k);
                    L(i, k) = mult;
                    for (std::size_t j = k; j < n; ++j)
                        U(i, j) -= mult * U(k, j);
                }
            }

            return std::make_tuple(L, U, perm);
        }

        // Gauss-Jacobi iterative method.
        // x0 may be provided; otherwise a zero vector is used.
        // Converges when infinity-norm of difference <= tol or maxIters reached.
        template <typename U>
        static Vector<typename std::common_type<T, U, double>::type>
            GaussJacobiIteration(const Matrix<T>& A, const Vector<U>& b,
                Vector<typename std::common_type<T, U, double>::type> x0 = Vector<typename std::common_type<T, U, double>::type>(),
                double tol = 1e-10, std::size_t maxIters = 1000)
        {
            using R = typename std::common_type<T, U, double>::type;
            if (!A.is_square()) throw std::runtime_error("GaussJacobiIteration: A must be square");
            std::size_t n = A.rows();
            if (b.size() != n) throw std::runtime_error("GaussJacobiIteration: size mismatch");

            Vector<R> x(n);
            if (x0.size() == n) x = x0;
            else x = Vector<R>(n, R());

            Vector<R> xnew(n);

            for (std::size_t iter = 0; iter < maxIters; ++iter)
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    R sum = R();
                    R diag = static_cast<R>(A(i, i));
                    if (std::abs(diag) <= std::numeric_limits<R>::epsilon())
                        throw std::runtime_error("GaussJacobiIteration: zero on diagonal");
                    for (std::size_t j = 0; j < n; ++j)
                    {
                        if (j == i) continue;
                        sum += static_cast<R>(A(i, j)) * x[j];
                    }
                    xnew[i] = (static_cast<R>(b[i]) - sum) / diag;
                }

                // check convergence (infinity norm)
                R maxdiff = R();
                for (std::size_t i = 0; i < n; ++i) maxdiff = std::max(maxdiff, std::abs(xnew[i] - x[i]));
                x = xnew;
                if (maxdiff <= static_cast<R>(tol)) break;
            }
            return x;
        }

        // Gauss-Seidel iterative method (in-place updates).
        template <typename U>
        static Vector<typename std::common_type<T, U, double>::type>
            GaussSeidelIteration(const Matrix<T>& A, const Vector<U>& b,
                Vector<typename std::common_type<T, U, double>::type> x0 = Vector<typename std::common_type<T, U, double>::type>(),
                double tol = 1e-10, std::size_t maxIters = 1000)
        {
            using R = typename std::common_type<T, U, double>::type;
            if (!A.is_square()) throw std::runtime_error("GaussSeidelIteration: A must be square");
            std::size_t n = A.rows();
            if (b.size() != n) throw std::runtime_error("GaussSeidelIteration: size mismatch");

            Vector<R> x(n);
            if (x0.size() == n) x = x0;
            else x = Vector<R>(n, R());

            for (std::size_t iter = 0; iter < maxIters; ++iter)
            {
                R maxdiff = R();
                for (std::size_t i = 0; i < n; ++i)
                {
                    R sum = R();
                    R diag = static_cast<R>(A(i, i));
                    if (std::abs(diag) <= std::numeric_limits<R>::epsilon())
                        throw std::runtime_error("GaussSeidelIteration: zero on diagonal");
                    for (std::size_t j = 0; j < n; ++j)
                    {
                        if (j == i) continue;
                        sum += static_cast<R>(A(i, j)) * x[j];
                    }
                    R x_new = (static_cast<R>(b[i]) - sum) / diag;
                    maxdiff = std::max(maxdiff, std::abs(x_new - x[i]));
                    x[i] = x_new;
                }
                if (maxdiff <= static_cast<R>(tol)) break;
            }
            return x;
        }

        // Jacobi eigenvalue algorithm for real symmetric matrices.
        // Returns pair (eigenvalues, eigenvectors) where eigenvectors are columns of the matrix.
        static std::pair< Vector<typename std::common_type<T, double>::type>,
            Matrix<typename std::common_type<T, double>::type> >
            JacobiEigen(const Matrix<T>& A, double tol = 1e-10, std::size_t maxIters = 1000)
        {
            using R = typename std::common_type<T, double>::type;
            if (!A.is_square()) throw std::runtime_error("JacobiEigen: A must be square");
            std::size_t n = A.rows();

            // check symmetry
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    if (std::abs(static_cast<R>(A(i, j)) - static_cast<R>(A(j, i))) > std::sqrt(std::numeric_limits<R>::epsilon()))
                        throw std::runtime_error("JacobiEigen: matrix must be symmetric");

            // working copy
            std::vector<std::vector<R>> a(n, std::vector<R>(n));
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    a[i][j] = static_cast<R>(A(i, j));

            Matrix<R> V = Matrix<R>::Eye(n);

            for (std::size_t iter = 0; iter < maxIters; ++iter)
            {
                // find largest off-diagonal element
                std::size_t p = 0, q = 1;
                R maxoff = std::abs(a[p][q]);
                for (std::size_t i = 0; i < n; ++i)
                {
                    for (std::size_t j = i + 1; j < n; ++j)
                    {
                        R v = std::abs(a[i][j]);
                        if (v > maxoff) { maxoff = v; p = i; q = j; }
                    }
                }

                if (maxoff <= static_cast<R>(tol)) break;

                // compute rotation angle
                R app = a[p][p];
                R aqq = a[q][q];
                R apq = a[p][q];

                R theta = 0.5L * std::atan2(static_cast<long double>(2.0L * apq), static_cast<long double>(aqq - app));
                R c = std::cos(theta);
                R s = std::sin(theta);

                // apply rotation to matrix a
                // update diagonal elements
                R app1 = c * c * app - 2.0L * s * c * apq + s * s * aqq;
                R aqq1 = s * s * app + 2.0L * s * c * apq + c * c * aqq;
                a[p][p] = app1;
                a[q][q] = aqq1;
                a[p][q] = a[q][p] = R(0);

                // update other elements
                for (std::size_t r = 0; r < n; ++r)
                {
                    if (r == p || r == q) continue;
                    R arp = a[r][p];
                    R arq = a[r][q];
                    a[r][p] = a[p][r] = c * arp - s * arq;
                    a[r][q] = a[q][r] = s * arp + c * arq;
                }

                // accumulate eigenvector rotations in V (columns)
                for (std::size_t r = 0; r < n; ++r)
                {
                    R vrp = V(r, p);
                    R vrq = V(r, q);
                    V(r, p) = c * vrp - s * vrq;
                    V(r, q) = s * vrp + c * vrq;
                }
            }

            Vector<R> evals(n);
            for (std::size_t i = 0; i < n; ++i) evals[i] = a[i][i];

            return std::make_pair(evals, V);
        }
    };

} // namespace CNum

#endif // NUMERICALS_LINEAREQUATIONS_H