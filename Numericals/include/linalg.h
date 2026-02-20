#ifndef NUMERICALS_LINALG_H
#define NUMERICALS_LINALG_H

#include "vector.h"
#include "matrix.h"
#include <type_traits>

namespace CNum
{

    // Dot product of two vectors (length must match)
    template <typename T>
    T dot(const Vector<T>& a, const Vector<T>& b)
    {
        if (a.size() != b.size())
            throw std::runtime_error("dot: vector size mismatch");
        T acc = T();
        for (std::size_t i = 0; i < a.size(); ++i)
            acc += a[i] * b[i];
        return acc;
    }

    // Matrix-vector multiply y = A * x
    template <typename T>
    Vector<T> matmul(const Matrix<T>& A, const Vector<T>& x)
    {
        std::size_t r = A.rows();
        std::size_t c = A.cols();
        if (c != x.size())
            throw std::runtime_error("matmul(matrix, vector): dimensions incompatible");
        Vector<T> y(r, T());
        for (std::size_t i = 0; i < r; ++i)
        {
            T acc = T();
            for (std::size_t j = 0; j < c; ++j)
                acc += A(i, j) * x[j];
            y[i] = acc;
        }
        return y;
    }

    // Matrix-matrix multiply C = A * B (naive implementation)
    template <typename T>
    Matrix<T> matmul(const Matrix<T>& A, const Matrix<T>& B)
    {
        std::size_t ar = A.rows();
        std::size_t ac = A.cols();
        std::size_t br = B.rows();
        std::size_t bc = B.cols();
        if (ac != br)
            throw std::runtime_error("matmul(matrix, matrix): dimensions incompatible");
        Matrix<T> C(ar, bc, T());
        for (std::size_t i = 0; i < ar; ++i)
        {
            for (std::size_t k = 0; k < ac; ++k)
            {
                T aik = A(i, k);
                for (std::size_t j = 0; j < bc; ++j)
                {
                    C(i, j) = C(i, j) + aik * B(k, j);
                }
            }
        }
        return C;
    }

    // Transpose
    template <typename T>
    Matrix<T> transpose(const Matrix<T>& A)
    {
        Matrix<T> B(A.cols(), A.rows());
        for (std::size_t i = 0; i < A.rows(); ++i)
            for (std::size_t j = 0; j < A.cols(); ++j)
                B(j, i) = A(i, j);
        return B;
    }

    // Implement Matrix * Vector and Matrix * Matrix operator overloads here
    // so matmul definitions are visible at the point of instantiation.
    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator*(const Matrix<T>& A, const Vector<U>& x)
    {
        using R = typename std::common_type<T, U>::type;
        auto nA = carray_cast<R>(A.to_carray());
        auto nx = carray_cast<R>(x.to_carray());
        Matrix<R> mA(std::move(nA));
        Vector<R> vx(std::move(nx));
        return matmul(mA, vx);
    }

    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator*(const Matrix<T>& A, const Matrix<U>& B)
    {
        using R = typename std::common_type<T, U>::type;
        auto nA = carray_cast<R>(A.to_carray());
        auto nB = carray_cast<R>(B.to_carray());
        Matrix<R> mA(std::move(nA));
        Matrix<R> mB(std::move(nB));
        return matmul(mA, mB);
    }

} // namespace CNum

#endif // NUMERICALS_LINALG_H