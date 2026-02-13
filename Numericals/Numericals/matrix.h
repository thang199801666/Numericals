#ifndef NUMERICALS_MATRIX_H
#define NUMERICALS_MATRIX_H

#include "ndarray.h"
#include "vector.h"
#include <initializer_list>
#include <type_traits>
#include <stdexcept>
#include <cmath>
#include <utility>
#include <limits>
#include <algorithm>

namespace CNum
{

    // NOTE: internal template parameter named Ty to avoid MSVC parsing ambiguity
    template <typename Ty>
    class Matrix
    {
    public:
        using value_type = Ty;

        // element reference types derived from underlying NdArray
        using elem_ref = decltype(std::declval<NdArray<Ty>&>()[0]);
        using const_elem_ref = decltype(std::declval<const NdArray<Ty>&>()[0]);

        // Transpose accessor so users can write `m.T` (numpy-style) and get a Matrix<Ty> copy.
        // It holds a pointer to the owning Matrix and converts to Matrix<Ty> via Transpose().
        class TransposeProxy
        {
        public:
            TransposeProxy(Matrix<Ty>* m = nullptr) : m_(m) {}
            // conversion to Matrix<Ty> triggers a copy of the transposed matrix
            operator Matrix<Ty>() const
            {
                if (!m_) throw std::runtime_error("TransposeProxy: null owner");
                return m_->Transpose();
            }
            // allow resetting the bound owner (used by assignment)
            void bind(Matrix<Ty>* m) { m_ = m; }

        private:
            Matrix<Ty>* m_;
        };

        // public member so code can use `m.T`
        TransposeProxy T;

        Matrix()
            : arr_(shape_t{}), T(this)
        {
        }

        Matrix(std::size_t rows, std::size_t cols)
            : arr_(shape_t{ rows, cols }), T(this)
        {
        }

        Matrix(std::size_t rows, std::size_t cols, const Ty& fill_value)
            : arr_(shape_t{ rows, cols }, fill_value), T(this)
        {
        }

        // initializer list of rows: { {1,2}, {3,4} }
        template <typename U = Ty>
        Matrix(std::initializer_list<std::initializer_list<U>> rows)
            : arr_(shape_t{}), T(this)
        {
            std::size_t r = rows.size();
            std::size_t c = 0;
            if (r > 0)
                c = rows.begin()->size();
            arr_ = NdArray<Ty>(shape_t{ r, c });
            std::size_t i = 0;
            for (auto const& row : rows)
            {
                if (row.size() != c)
                    throw std::runtime_error("Matrix: inconsistent row lengths in initializer");
                std::size_t j = 0;
                for (auto const& v : row)
                {
                    arr_.at({ i, j }) = static_cast<Ty>(v);
                    ++j;
                }
                ++i;
            }
        }

        // construct from NdArray (expects 2-D)
        explicit Matrix(NdArray<Ty>&& a)
            : arr_(std::move(a)), T(this)
        {
            if (arr_.shape().size() != 2)
                throw std::runtime_error("Matrix: underlying NdArray must be 2-D");
        }

        // copy constructor: ensure proxy binds to the new object
        Matrix(const Matrix& other)
            : arr_(other.arr_), T(this)
        {
        }

        // move constructor: ensure proxy binds to the new object
        Matrix(Matrix&& other) noexcept
            : arr_(std::move(other.arr_)), T(this)
        {
        }

        // copy assignment
        Matrix& operator=(const Matrix& other)
        {
            if (this != &other)
            {
                arr_ = other.arr_;
                T.bind(this);
            }
            return *this;
        }

        // move assignment
        Matrix& operator=(Matrix&& other) noexcept
        {
            if (this != &other)
            {
                arr_ = std::move(other.arr_);
                T.bind(this);
            }
            return *this;
        }

        std::size_t rows() const noexcept { return arr_.shape().empty() ? 0 : arr_.shape()[0]; }
        std::size_t cols() const noexcept { return (arr_.shape().size() < 2) ? 0 : arr_.shape()[1]; }

        // element access forwards the NdArray element reference type
        auto operator()(std::size_t r, std::size_t c) -> elem_ref
        {
            return arr_.at({ r, c });
        }
        auto operator()(std::size_t r, std::size_t c) const -> const_elem_ref
        {
            return arr_.at({ r, c });
        }

        NdArray<Ty> to_ndarray() const { return arr_; }
        // Backwards-compatible alias (preserve existing name)
        NdArray<Ty> to_ndarray_alias() const { return to_ndarray(); }

        //
        // Row / Column view proxies to allow m[i][j], assign whole row/column from Vector/NdArray/initializer_list
        //
        class RowProxy
        {
        public:
            RowProxy(Matrix<Ty>& m, std::size_t r) : m_(m), r_(r) {}
            auto operator[](std::size_t c) -> elem_ref { return m_.operator()(r_, c); }
            auto operator[](std::size_t c) const -> const_elem_ref { return m_.operator()(r_, c); }

            // assign whole row from Vector<U>
            template <typename U>
            RowProxy& operator=(const Vector<U>& v)
            {
                if (m_.cols() != v.size())
                    throw std::runtime_error("Row assignment: size mismatch");
                for (std::size_t c = 0; c < m_.cols(); ++c)
                    m_.operator()(r_, c) = static_cast<Ty>(v[c]);
                return *this;
            }

            // assign from 1-D NdArray<U>
            template <typename U>
            RowProxy& operator=(const NdArray<U>& a)
            {
                if (a.shape().size() != 1)
                    throw std::runtime_error("Row assignment: source must be 1-D");
                if (m_.cols() != a.size())
                    throw std::runtime_error("Row assignment: size mismatch");
                for (std::size_t c = 0; c < m_.cols(); ++c)
                    m_.operator()(r_, c) = static_cast<Ty>(a[c]);
                return *this;
            }

            // assign from initializer_list
            template <typename U>
            RowProxy& operator=(std::initializer_list<U> list)
            {
                if (m_.cols() != list.size())
                    throw std::runtime_error("Row assignment: size mismatch");
                std::size_t c = 0;
                for (auto const& v : list)
                {
                    m_.operator()(r_, c++) = static_cast<Ty>(v);
                }
                return *this;
            }

        private:
            Matrix<Ty>& m_;
            std::size_t r_;
        };

        class ConstRowProxy
        {
        public:
            ConstRowProxy(const Matrix<Ty>& m, std::size_t r) : m_(m), r_(r) {}
            auto operator[](std::size_t c) const -> const_elem_ref { return m_.operator()(r_, c); }
        private:
            const Matrix<Ty>& m_;
            std::size_t r_;
        };

        class ColProxy
        {
        public:
            ColProxy(Matrix<Ty>& m, std::size_t c) : m_(m), c_(c) {}
            auto operator[](std::size_t r) -> elem_ref { return m_.operator()(r, c_); }
            auto operator[](std::size_t r) const -> const_elem_ref { return m_.operator()(r, c_); }

            // assign whole column from Vector<U>
            template <typename U>
            ColProxy& operator=(const Vector<U>& v)
            {
                if (m_.rows() != v.size())
                    throw std::runtime_error("Column assignment: size mismatch");
                for (std::size_t r = 0; r < m_.rows(); ++r)
                    m_.operator()(r, c_) = static_cast<Ty>(v[r]);
                return *this;
            }

            // assign from 1-D NdArray<U>
            template <typename U>
            ColProxy& operator=(const NdArray<U>& a)
            {
                if (a.shape().size() != 1)
                    throw std::runtime_error("Column assignment: source must be 1-D");
                if (m_.rows() != a.size())
                    throw std::runtime_error("Column assignment: size mismatch");
                for (std::size_t r = 0; r < m_.rows(); ++r)
                    m_.operator()(r, c_) = static_cast<Ty>(a[r]);
                return *this;
            }

            // assign from initializer_list
            template <typename U>
            ColProxy& operator=(std::initializer_list<U> list)
            {
                if (m_.rows() != list.size())
                    throw std::runtime_error("Column assignment: size mismatch");
                std::size_t r = 0;
                for (auto const& v : list)
                {
                    m_.operator()(r++, c_) = static_cast<Ty>(v);
                }
                return *this;
            }

        private:
            Matrix<Ty>& m_;
            std::size_t c_;
        };

        class ConstColProxy
        {
        public:
            ConstColProxy(const Matrix<Ty>& m, std::size_t c) : m_(m), c_(c) {}
            auto operator[](std::size_t r) const -> const_elem_ref { return m_.operator()(r, c_); }
        private:
            const Matrix<Ty>& m_;
            std::size_t c_;
        };

        // operator[] returns a row proxy so m[i][j] works
        RowProxy operator[](std::size_t r)
        {
            if (r >= rows()) throw std::out_of_range("row index out of range");
            return RowProxy(*this, r);
        }

        ConstRowProxy operator[](std::size_t r) const
        {
            if (r >= rows()) throw std::out_of_range("row index out of range");
            return ConstRowProxy(*this, r);
        }

        // explicit row/col accessors for assigning entire row/column:
        RowProxy row(std::size_t r) { return (*this)[r]; }
        ConstRowProxy row(std::size_t r) const { return (*this)[r]; }

        ColProxy col(std::size_t c)
        {
            if (c >= cols()) throw std::out_of_range("column index out of range");
            return ColProxy(*this, c);
        }

        ConstColProxy col(std::size_t c) const
        {
            if (c >= cols()) throw std::out_of_range("column index out of range");
            return ConstColProxy(*this, c);
        }

        // Trace of matrix (sum of diagonal elements) - PascalCase public API
        auto Trace() const -> typename std::common_type<Ty, double>::type
        {
            using R = typename std::common_type<Ty, double>::type;
            R sum = R();
            std::size_t n = std::min(rows(), cols());
            for (std::size_t i = 0; i < n; ++i)
                sum += static_cast<R>((*this)(i, i));
            return sum;
        }
        // backwards-compatible lowercase alias
        auto trace() const -> decltype(this->Trace()) { return Trace(); }

        // Transpose - PascalCase public API
        Matrix<Ty> Transpose() const
        {
            Matrix<Ty> t(cols(), rows());
            for (std::size_t i = 0; i < rows(); ++i)
                for (std::size_t j = 0; j < cols(); ++j)
                    t(j, i) = (*this)(i, j);
            return t;
        }
        // backwards-compatible lowercase alias
        Matrix<Ty> transpose() const { return Transpose(); }

        // another alias for clarity (PascalCase)
        Matrix<Ty> Transposed() const
        {
            return Transpose();
        }
        // backwards-compatible lowercase alias
        Matrix<Ty> transposed() const { return Transposed(); }

        // Determinant via Gaussian elimination with partial pivoting - PascalCase public API
        auto Det() const -> typename std::common_type<Ty, double>::type
        {
            using R = typename std::common_type<Ty, double>::type;
            std::size_t n = rows();
            if (n != cols())
                throw std::runtime_error("Matrix::Det requires square matrix");

            if (n == 0)
                return R(1); // determinant of empty matrix considered 1

            // copy into a mutable 2D container with floating type R
            std::vector<std::vector<R>> a(n, std::vector<R>(n));
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    a[i][j] = static_cast<R>((*this)(i, j));

            int sign = 1;

            for (std::size_t i = 0; i < n; ++i)
            {
                // partial pivot: find row with largest absolute value in column i
                std::size_t pivot = i;
                R maxabs = std::abs(a[i][i]);
                for (std::size_t r = i + 1; r < n; ++r)
                {
                    R val = std::abs(a[r][i]);
                    if (val > maxabs)
                    {
                        maxabs = val;
                        pivot = r;
                    }
                }

                // if pivot element is (near) zero, determinant is zero
                if (std::abs(a[pivot][i]) <= std::numeric_limits<R>::epsilon() * R(1e2))
                    return R(0);

                if (pivot != i)
                {
                    std::swap(a[pivot], a[i]);
                    sign = -sign;
                }

                // eliminate below
                for (std::size_t r = i + 1; r < n; ++r)
                {
                    R factor = a[r][i] / a[i][i];
                    for (std::size_t c = i; c < n; ++c)
                        a[r][c] -= factor * a[i][c];
                }
            }

            // product of diagonal
            R determinant = (sign == 1) ? R(1) : R(-1);
            for (std::size_t i = 0; i < n; ++i)
                determinant *= a[i][i];

            return determinant;
        }
        // backwards-compatible lowercase alias
        auto det() const -> decltype(this->Det()) { return Det(); }

    private:
        NdArray<Ty> arr_;
    };

    // forward-declare matmul signatures to avoid circular includes with linalg.h
    template <typename T> Vector<T> matmul(const Matrix<T>& A, const Vector<T>& x);
    template <typename T> Matrix<T> matmul(const Matrix<T>& A, const Matrix<T>& B);

    // Elementwise (Hadamard) product helper — explicit name to avoid ambiguity with linear algebra *
    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> hadamard(const Matrix<T>& a, const Matrix<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = ndarray_cast<R>(a.to_ndarray());
        auto nb = ndarray_cast<R>(b.to_ndarray());
        // construct Matrix<R> directly from NdArray<R>
        return Matrix<R>(CNum::mul(std::move(na), std::move(nb)));
    }

    // elementwise matrix ops via ufuncs (kept for add/sub/div) with mixed types
    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator+(const Matrix<T>& a, const Matrix<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = ndarray_cast<R>(a.to_ndarray());
        auto nb = ndarray_cast<R>(b.to_ndarray());
        return Matrix<R>(CNum::add(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator-(const Matrix<T>& a, const Matrix<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = ndarray_cast<R>(a.to_ndarray());
        auto nb = ndarray_cast<R>(b.to_ndarray());
        return Matrix<R>(CNum::sub(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator/(const Matrix<T>& a, const Matrix<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = ndarray_cast<R>(a.to_ndarray());
        auto nb = ndarray_cast<R>(b.to_ndarray());
        return Matrix<R>(CNum::div(std::move(na), std::move(nb)));
    }

    // Matrix * Vector -> matrix-vector multiply (linear algebra)
    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator*(const Matrix<T>& A, const Vector<U>& x)
    {
        using R = typename std::common_type<T, U>::type;
        auto nA = ndarray_cast<R>(A.to_ndarray());
        auto nx = ndarray_cast<R>(x.to_ndarray());
        // construct Matrix<R> and Vector<R> from NdArray<R> and call matmul
        Matrix<R> mA(std::move(nA));
        Vector<R> vx(std::move(nx));
        return matmul(mA, vx);
    }

    // Matrix * Matrix -> matrix-matrix multiply (linear algebra) with mixed types
    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator*(const Matrix<T>& A, const Matrix<U>& B)
    {
        using R = typename std::common_type<T, U>::type;
        auto nA = ndarray_cast<R>(A.to_ndarray());
        auto nB = ndarray_cast<R>(B.to_ndarray());
        Matrix<R> mA(std::move(nA));
        Matrix<R> mB(std::move(nB));
        return matmul(mA, mB);
    }

    // Scalar ops for Matrix (scalar on right and left) with mixed scalar type
    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator+(const Matrix<T>& m, const U& scalar)
    {
        using R = typename std::common_type<T, U>::type;
        Matrix<R> r(ndarray_cast<R>(m.to_ndarray()));
        for (std::size_t i = 0; i < r.rows(); ++i)
            for (std::size_t j = 0; j < r.cols(); ++j)
                r(i, j) = r(i, j) + static_cast<R>(scalar);
        return r;
    }

    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator+(const U& scalar, const Matrix<T>& m) { return m + scalar; }

    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator-(const Matrix<T>& m, const U& scalar)
    {
        using R = typename std::common_type<T, U>::type;
        Matrix<R> r(ndarray_cast<R>(m.to_ndarray()));
        for (std::size_t i = 0; i < r.rows(); ++i)
            for (std::size_t j = 0; j < r.cols(); ++j)
                r(i, j) = r(i, j) - static_cast<R>(scalar);
        return r;
    }

    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator-(const U& scalar, const Matrix<T>& m)
    {
        using R = typename std::common_type<T, U>::type;
        Matrix<R> r(m.rows(), m.cols());
        for (std::size_t i = 0; i < m.rows(); ++i)
            for (std::size_t j = 0; j < m.cols(); ++j)
                r(i, j) = static_cast<R>(scalar) - static_cast<R>(m(i, j));
        return r;
    }

    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator*(const Matrix<T>& m, const U& scalar)
    {
        using R = typename std::common_type<T, U>::type;
        Matrix<R> r(ndarray_cast<R>(m.to_ndarray()));
        for (std::size_t i = 0; i < r.rows(); ++i)
            for (std::size_t j = 0; j < r.cols(); ++j)
                r(i, j) = r(i, j) * static_cast<R>(scalar);
        return r;
    }

    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator*(const U& scalar, const Matrix<T>& m) { return m * scalar; }

    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator/(const Matrix<T>& m, const U& scalar)
    {
        using R = typename std::common_type<T, U>::type;
        Matrix<R> r(ndarray_cast<R>(m.to_ndarray()));
        for (std::size_t i = 0; i < r.rows(); ++i)
            for (std::size_t j = 0; j < r.cols(); ++j)
                r(i, j) = r(i, j) / static_cast<R>(scalar);
        return r;
    }

    // Compound assignment with scalar for convenience (in-place only for same-type scalar)
    template <typename T>
    Matrix<T>& operator+=(Matrix<T>& m, const T& scalar)
    {
        for (std::size_t i = 0; i < m.rows(); ++i)
            for (std::size_t j = 0; j < m.cols(); ++j)
                m(i, j) = m(i, j) + scalar;
        return m;
    }

    template <typename T>
    Matrix<T>& operator-=(Matrix<T>& m, const T& scalar)
    {
        for (std::size_t i = 0; i < m.rows(); ++i)
            for (std::size_t j = 0; j < m.cols(); ++j)
                m(i, j) = m(i, j) - scalar;
        return m;
    }

    template <typename T>
    Matrix<T>& operator*=(Matrix<T>& m, const T& scalar)
    {
        for (std::size_t i = 0; i < m.rows(); ++i)
            for (std::size_t j = 0; j < m.cols(); ++j)
                m(i, j) = m(i, j) * scalar;
        return m;
    }

    template <typename T>
    Matrix<T>& operator/=(Matrix<T>& m, const T& scalar)
    {
        for (std::size_t i = 0; i < m.rows(); ++i)
            for (std::size_t j = 0; j < m.cols(); ++j)
                m(i, j) = m(i, j) / scalar;
        return m;
    }

} // namespace CNum

#endif // NUMERICALS_MATRIX_H