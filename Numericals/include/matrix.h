#ifndef NUMERICALS_MATRIX_H
#define NUMERICALS_MATRIX_H

#include "export.h"
#include "CArray.h"
#include "vector.h"
#include <initializer_list>
#include <type_traits>
#include <stdexcept>
#include <cmath>
#include <utility>
#include <limits>
#include <algorithm>
#include <ostream>

namespace CNum
{

    // NOTE: internal template parameter named Ty to avoid MSVC parsing ambiguity
    template <typename Ty>
    class NUMERICALS_API Matrix
    {
    public:
        using value_type = Ty;

        // element reference types derived from underlying CArray
        using elem_ref = decltype(std::declval<CArray<Ty>&>()[0]);
        using const_elem_ref = decltype(std::declval<const CArray<Ty>&>()[0]);

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
            arr_ = CArray<Ty>(shape_t{ r, c });
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

        // construct from CArray (expects 2-D)
        explicit Matrix(CArray<Ty>&& a)
            : arr_(std::move(a)), T(this)
        {
            if (arr_.shape().size() != 2)
                throw std::runtime_error("Matrix: underlying CArray must be 2-D");
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

        // element access forwards the CArray element reference type
        auto operator()(std::size_t r, std::size_t c) -> elem_ref
        {
            return arr_.at({ r, c });
        }
        auto operator()(std::size_t r, std::size_t c) const -> const_elem_ref
        {
            return arr_.at({ r, c });
        }

        // provide direct access to underlying contiguous storage for hot loops
        Ty* data() noexcept { return arr_.data(); }
        const Ty* data() const noexcept { return arr_.data(); }

        // provide const-ref to underlying CArray to avoid copies when caller only needs read access
        const CArray<Ty>& carray_ref() const noexcept { return arr_; }
        CArray<Ty>& carray_ref() noexcept { return arr_; }

        // Backwards-compatible conversion that returns a copy of the underlying CArray
        CArray<Ty> to_carray() const { return arr_; }
        // Backwards-compatible alias (preserve existing name if needed)
        CArray<Ty> to_carray_alias() const { return to_carray(); }

        //
        // Row / Column view proxies to allow m[i][j], assign whole row/column from Vector/CArray/initializer_list
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

            // assign from 1-D CArray<U>
            template <typename U>
            RowProxy& operator=(const CArray<U>& a)
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

            // assign from 1-D CArray<U>
            template <typename U>
            ColProxy& operator=(const CArray<U>& a)
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

        //
        // Additional "full matrix" convenience methods
        //

        // Factories
        static Matrix<Ty> Zeros(std::size_t r, std::size_t c)
        {
            return Matrix<Ty>(r, c, Ty());
        }

        static Matrix<Ty> Ones(std::size_t r, std::size_t c)
        {
            return Matrix<Ty>(r, c, static_cast<Ty>(1));
        }

        static Matrix<Ty> Eye(std::size_t n)
        {
            Matrix<Ty> m(n, n, Ty());
            for (std::size_t i = 0; i < n; ++i)
                m(i, i) = static_cast<Ty>(1);
            return m;
        }

        // Fill in-place
        void Fill(const Ty& value)
        {
            for (std::size_t i = 0; i < rows(); ++i)
                for (std::size_t j = 0; j < cols(); ++j)
                    (*this)(i, j) = value;
        }

        // Basic queries
        bool is_empty() const noexcept { return rows() == 0 || cols() == 0; }
        bool is_square() const noexcept { return rows() == cols(); }
        std::pair<std::size_t, std::size_t> shape() const noexcept { return { rows(), cols() }; }

        // elementwise sum and mean
        auto Sum() const -> typename std::common_type<Ty, double>::type
        {
            using R = typename std::common_type<Ty, double>::type;
            R s = R();
            for (std::size_t i = 0; i < rows(); ++i)
                for (std::size_t j = 0; j < cols(); ++j)
                    s += static_cast<R>((*this)(i, j));
            return s;
        }
        auto sum() const -> decltype(this->Sum()) { return Sum(); }

        // axis-aware Sum: axis == 0 -> column sums (length cols)
        //                  axis == 1 -> row sums (length rows)
        auto Sum(int axis) const -> Vector<typename std::common_type<Ty, double>::type>
        {
            using R = typename std::common_type<Ty, double>::type;
            if (axis == 0)
            {
                Vector<R> v(cols());
                for (std::size_t j = 0; j < cols(); ++j)
                {
                    R s = R();
                    for (std::size_t i = 0; i < rows(); ++i)
                        s += static_cast<R>((*this)(i, j));
                    v[j] = s;
                }
                return v;
            }
            else if (axis == 1)
            {
                Vector<R> v(rows());
                for (std::size_t i = 0; i < rows(); ++i)
                {
                    R s = R();
                    for (std::size_t j = 0; j < cols(); ++j)
                        s += static_cast<R>((*this)(i, j));
                    v[i] = s;
                }
                return v;
            }
            else
            {
                throw std::runtime_error("Sum: invalid axis (must be 0 or 1)");
            }
        }
        auto sum(int axis) const -> decltype(this->Sum(axis)) { return Sum(axis); }

        auto Mean() const -> typename std::common_type<Ty, double>::type
        {
            using R = typename std::common_type<Ty, double>::type;
            if (is_empty()) return R();
            return Sum() / static_cast<R>(rows() * cols());
        }
        auto mean() const -> decltype(this->Mean()) { return Mean(); }

        // axis-aware Mean: axis == 0 -> column means (length cols)
        //                  axis == 1 -> row means (length rows)
        auto Mean(int axis) const -> Vector<typename std::common_type<Ty, double>::type>
        {
            using R = typename std::common_type<Ty, double>::type;
            if (axis == 0)
            {
                if (cols() == 0) return Vector<R>(0);
                Vector<R> v(cols());
                for (std::size_t j = 0; j < cols(); ++j)
                {
                    R s = R();
                    for (std::size_t i = 0; i < rows(); ++i)
                        s += static_cast<R>((*this)(i, j));
                    v[j] = s / static_cast<R>(rows());
                }
                return v;
            }
            else if (axis == 1)
            {
                if (rows() == 0) return Vector<R>(0);
                Vector<R> v(rows());
                for (std::size_t i = 0; i < rows(); ++i)
                {
                    R s = R();
                    for (std::size_t j = 0; j < cols(); ++j)
                        s += static_cast<R>((*this)(i, j));
                    v[i] = s / static_cast<R>(cols());
                }
                return v;
            }
            else
            {
                throw std::runtime_error("Mean: invalid axis (must be 0 or 1)");
            }
        }
        auto mean(int axis) const -> decltype(this->Mean(axis)) { return Mean(axis); }

        // Frobenius norm
        auto FrobeniusNorm() const -> typename std::common_type<Ty, double>::type
        {
            using R = typename std::common_type<Ty, double>::type;
            R s = R();
            for (std::size_t i = 0; i < rows(); ++i)
                for (std::size_t j = 0; j < cols(); ++j)
                {
                    R v = static_cast<R>((*this)(i, j));
                    s += v * v;
                }
            return std::sqrt(s);
        }
        auto frobenius_norm() const -> decltype(this->FrobeniusNorm()) { return FrobeniusNorm(); }

        // extract diagonal as Vector<R>
        template <typename U = Ty>
        Vector<typename std::common_type<U, double>::type> Diagonal() const
        {
            using R = typename std::common_type<U, double>::type;
            std::size_t n = std::min(rows(), cols());
            Vector<R> v(n);
            for (std::size_t i = 0; i < n; ++i)
                v[i] = static_cast<R>((*this)(i, i));
            return v;
        }
        template <typename U = Ty>
        Vector<typename std::common_type<U, double>::type> diagonal() const { return Diagonal<U>(); }

        // set diagonal from Vector
        template <typename U>
        void SetDiagonal(const Vector<U>& diag)
        {
            std::size_t n = std::min(rows(), cols());
            if (diag.size() != n)
                throw std::runtime_error("SetDiagonal: size mismatch");
            for (std::size_t i = 0; i < n; ++i)
                (*this)(i, i) = static_cast<Ty>(diag[i]);
        }
        template <typename U>
        void set_diagonal(const Vector<U>& diag) { SetDiagonal(diag); }

        // set diagonal from Vector
        template <typename U>
        void setDiagonal(const Vector<U>& diag) { SetDiagonal(diag); }

        // Flatten / Ravel -> 1-D Vector<Ty>
        Vector<Ty> Flatten() const
        {
            Vector<Ty> v(rows() * cols());
            std::size_t idx = 0;
            for (std::size_t i = 0; i < rows(); ++i)
                for (std::size_t j = 0; j < cols(); ++j)
                    v[idx++] = static_cast<Ty>((*this)(i, j));
            return v;
        }
        Vector<Ty> flatten() const { return Flatten(); }
        Vector<Ty> Ravel() const { return Flatten(); }
        Vector<Ty> ravel() const { return Ravel(); }

        // Reshape -> returns new Matrix with same elements in row-major order
        Matrix<Ty> Reshape(std::size_t new_rows, std::size_t new_cols) const
        {
            std::size_t total = rows() * cols();
            if (new_rows * new_cols != total)
                throw std::runtime_error("Reshape: total size must remain unchanged");
            Matrix<Ty> m(new_rows, new_cols);
            std::size_t idx = 0;
            for (std::size_t i = 0; i < new_rows; ++i)
            {
                for (std::size_t j = 0; j < new_cols; ++j)
                {
                    std::size_t src_i = idx / cols();
                    std::size_t src_j = idx % cols();
                    m(i, j) = (*this)(src_i, src_j);
                    ++idx;
                }
            }
            return m;
        }
        Matrix<Ty> reshape(std::size_t r, std::size_t c) const { return Reshape(r, c); }

        // argmax / argmin (global and axis)
        std::pair<std::size_t, std::size_t> ArgMax() const
        {
            using R = typename std::common_type<Ty, double>::type;
            if (is_empty()) throw std::runtime_error("ArgMax: matrix is empty");
            std::size_t mi = 0, mj = 0;
            R best = static_cast<R>((*this)(0, 0));
            for (std::size_t i = 0; i < rows(); ++i)
            {
                for (std::size_t j = 0; j < cols(); ++j)
                {
                    R v = static_cast<R>((*this)(i, j));
                    if (v > best)
                    {
                        best = v;
                        mi = i;
                        mj = j;
                    }
                }
            }
            return { mi, mj };
        }
        std::pair<std::size_t, std::size_t> argmax() const { return ArgMax(); }

        Vector<std::size_t> ArgMax(int axis) const
        {
            if (axis == 0)
            {
                Vector<std::size_t> v(cols());
                for (std::size_t j = 0; j < cols(); ++j)
                {
                    std::size_t best_i = 0;
                    auto best = (*this)(0, j);
                    for (std::size_t i = 1; i < rows(); ++i)
                        if (static_cast<typename std::common_type<Ty, double>::type>((*this)(i, j)) >
                            static_cast<typename std::common_type<Ty, double>::type>(best))
                        {
                            best = (*this)(i, j);
                            best_i = i;
                        }
                    v[j] = best_i;
                }
                return v;
            }
            else if (axis == 1)
            {
                Vector<std::size_t> v(rows());
                for (std::size_t i = 0; i < rows(); ++i)
                {
                    std::size_t best_j = 0;
                    auto best = (*this)(i, 0);
                    for (std::size_t j = 1; j < cols(); ++j)
                        if (static_cast<typename std::common_type<Ty, double>::type>((*this)(i, j)) >
                            static_cast<typename std::common_type<Ty, double>::type>(best))
                        {
                            best = (*this)(i, j);
                            best_j = j;
                        }
                    v[i] = best_j;
                }
                return v;
            }
            else
            {
                throw std::runtime_error("ArgMax: invalid axis (must be 0 or 1)");
            }
        }
        Vector<std::size_t> argmax(int axis) const { return ArgMax(axis); }

        std::pair<std::size_t, std::size_t> ArgMin() const
        {
            using R = typename std::common_type<Ty, double>::type;
            if (is_empty()) throw std::runtime_error("ArgMin: matrix is empty");
            std::size_t mi = 0, mj = 0;
            R best = static_cast<R>((*this)(0, 0));
            for (std::size_t i = 0; i < rows(); ++i)
            {
                for (std::size_t j = 0; j < cols(); ++j)
                {
                    R v = static_cast<R>((*this)(i, j));
                    if (v < best)
                    {
                        best = v;
                        mi = i;
                        mj = j;
                    }
                }
            }
            return { mi, mj };
        }
        std::pair<std::size_t, std::size_t> argmin() const { return ArgMin(); }

        Vector<std::size_t> ArgMin(int axis) const
        {
            if (axis == 0)
            {
                Vector<std::size_t> v(cols());
                for (std::size_t j = 0; j < cols(); ++j)
                {
                    std::size_t best_i = 0;
                    auto best = (*this)(0, j);
                    for (std::size_t i = 1; i < rows(); ++i)
                        if (static_cast<typename std::common_type<Ty, double>::type>((*this)(i, j)) <
                            static_cast<typename std::common_type<Ty, double>::type>(best))
                        {
                            best = (*this)(i, j);
                            best_i = i;
                        }
                    v[j] = best_i;
                }
                return v;
            }
            else if (axis == 1)
            {
                Vector<std::size_t> v(rows());
                for (std::size_t i = 0; i < rows(); ++i)
                {
                    std::size_t best_j = 0;
                    auto best = (*this)(i, 0);
                    for (std::size_t j = 1; j < cols(); ++j)
                        if (static_cast<typename std::common_type<Ty, double>::type>((*this)(i, j)) <
                            static_cast<typename std::common_type<Ty, double>::type>(best))
                        {
                            best = (*this)(i, j);
                            best_j = j;
                        }
                    v[i] = best_j;
                }
                return v;
            }
            else
            {
                throw std::runtime_error("ArgMin: invalid axis (must be 0 or 1)");
            }
        }
        Vector<std::size_t> argmin(int axis) const { return ArgMin(axis); }

        // swap rows and columns in-place
        void SwapRows(std::size_t r1, std::size_t r2)
        {
            if (r1 >= rows() || r2 >= rows()) throw std::out_of_range("SwapRows: index out of range");
            for (std::size_t c = 0; c < cols(); ++c)
                std::swap((*this)(r1, c), (*this)(r2, c));
        }
        void swap_rows(std::size_t r1, std::size_t r2) { SwapRows(r1, r2); }

        void SwapCols(std::size_t c1, std::size_t c2)
        {
            if (c1 >= cols() || c2 >= cols()) throw std::out_of_range("SwapCols: index out of range");
            for (std::size_t r = 0; r < rows(); ++r)
                std::swap((*this)(r, c1), (*this)(r, c2));
        }
        void swap_cols(std::size_t c1, std::size_t c2) { SwapCols(c1, c2); }

        // Unary negation
        Matrix<Ty> operator-() const
        {
            Matrix<Ty> r(rows(), cols());
            for (std::size_t i = 0; i < r.rows(); ++i)
                for (std::size_t j = 0; j < r.cols(); ++j)
                    r(i, j) = -(*this)(i, j);
            return r;
        }

        // equality operators (exact)
        bool operator==(const Matrix<Ty>& other) const
        {
            if (rows() != other.rows() || cols() != other.cols()) return false;
            for (std::size_t i = 0; i < rows(); ++i)
                for (std::size_t j = 0; j < cols(); ++j)
                    if (!((*this)(i, j) == other(i, j))) return false;
            return true;
        }
        bool operator!=(const Matrix<Ty>& other) const { return !(*this == other); }

        // Inverse using Gauss-Jordan (returns promoted floating-point matrix type)
        auto Inverse() const -> Matrix<typename std::common_type<Ty, double>::type>
        {
            using R = typename std::common_type<Ty, double>::type;
            if (!is_square())
                throw std::runtime_error("Inverse requires square matrix");
            std::size_t n = rows();
            if (n == 0) return Matrix<R>(0, 0);

            // augmented matrix [A | I]
            std::vector<std::vector<R>> a(n, std::vector<R>(2 * n));
            for (std::size_t i = 0; i < n; ++i)
            {
                for (std::size_t j = 0; j < n; ++j)
                    a[i][j] = static_cast<R>((*this)(i, j));
                for (std::size_t j = 0; j < n; ++j)
                    a[i][n + j] = (i == j) ? R(1) : R(0);
            }

            // Gauss-Jordan with partial pivoting
            for (std::size_t i = 0; i < n; ++i)
            {
                // find pivot
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
                if (std::abs(a[pivot][i]) <= std::numeric_limits<R>::epsilon() * R(1e2))
                    throw std::runtime_error("Inverse: matrix is singular or numerically unstable");

                if (pivot != i) std::swap(a[pivot], a[i]);

                // normalize pivot row
                R diag = a[i][i];
                for (std::size_t j = 0; j < 2 * n; ++j) a[i][j] /= diag;

                // eliminate other rows
                for (std::size_t r = 0; r < n; ++r)
                {
                    if (r == i) continue;
                    R factor = a[r][i];
                    for (std::size_t j = 0; j < 2 * n; ++j)
                        a[r][j] -= factor * a[i][j];
                }
            }

            // extract inverse
            Matrix<R> inv(n, n);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    inv(i, j) = a[i][n + j];

            return inv;
        }
        auto inverse() const -> decltype(this->Inverse()) { return Inverse(); }

        // Solve linear system Ax = b (returns promoted floating vector)
        template <typename U>
        Vector<typename std::common_type<Ty, U, double>::type> Solve(const Vector<U>& b) const
        {
            using R = typename std::common_type<Ty, U, double>::type;
            if (rows() != cols())
                throw std::runtime_error("Solve requires square matrix");
            std::size_t n = rows();
            if (b.size() != n)
                throw std::runtime_error("Solve: size mismatch");

            // Build augmented matrix with A and b
            std::vector<std::vector<R>> a(n, std::vector<R>(n + 1));
            for (std::size_t i = 0; i < n; ++i)
            {
                for (std::size_t j = 0; j < n; ++j)
                    a[i][j] = static_cast<R>((*this)(i, j));
                a[i][n] = static_cast<R>(b[i]);
            }

            // Gaussian elimination with partial pivoting
            for (std::size_t i = 0; i < n; ++i)
            {
                // pivot
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
                if (std::abs(a[pivot][i]) <= std::numeric_limits<R>::epsilon() * R(1e2))
                    throw std::runtime_error("Solve: matrix is singular or numerically unstable");

                if (pivot != i) std::swap(a[pivot], a[i]);

                // eliminate below
                for (std::size_t r = i + 1; r < n; ++r)
                {
                    R factor = a[r][i] / a[i][i];
                    for (std::size_t c = i; c <= n; ++c)
                        a[r][c] -= factor * a[i][c];
                }
            }

            // back substitution
            Vector<R> x(n);
            for (int i = static_cast<int>(n) - 1; i >= 0; --i)
            {
                R s = a[i][n];
                for (std::size_t j = i + 1; j < n; ++j)
                    s -= a[i][j] * x[j];
                x[i] = s / a[i][i];
            }
            return x;
        }
        template <typename U>
        Vector<typename std::common_type<Ty, U, double>::type> solve(const Vector<U>& b) const { return Solve(b); }

    private:
        CArray<Ty> arr_;
    };

    // forward-declare matmul signatures to avoid circular includes with linalg.h
    template <typename T> Vector<T> matmul(const Matrix<T>& A, const Vector<T>& x);
    template <typename T> Matrix<T> matmul(const Matrix<T>& A, const Matrix<T>& B);

    // Elementwise (Hadamard) product helper — explicit name to avoid ambiguity with linear algebra *
    // Optimized to avoid intermediate CArray copies: iterate directly and write into result
    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> hadamard(const Matrix<T>& a, const Matrix<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        if (a.rows() != b.rows() || a.cols() != b.cols())
            throw std::runtime_error("hadamard: shape mismatch");
        Matrix<R> out(a.rows(), a.cols());
        for (std::size_t i = 0; i < out.rows(); ++i)
            for (std::size_t j = 0; j < out.cols(); ++j)
                out(i, j) = static_cast<R>(a(i, j)) * static_cast<R>(b(i, j));
        return out;
    }

    // elementwise matrix ops via ufuncs (kept for add/sub/div) with mixed types
    // Implemented without copying underlying CArray to reduce temporaries.
    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator+(const Matrix<T>& a, const Matrix<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        if (a.rows() != b.rows() || a.cols() != b.cols())
            throw std::runtime_error("operator+: shape mismatch");
        Matrix<R> out(a.rows(), a.cols());
        for (std::size_t i = 0; i < out.rows(); ++i)
            for (std::size_t j = 0; j < out.cols(); ++j)
                out(i, j) = static_cast<R>(a(i, j)) + static_cast<R>(b(i, j));
        return out;
    }

    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator-(const Matrix<T>& a, const Matrix<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        if (a.rows() != b.rows() || a.cols() != b.cols())
            throw std::runtime_error("operator-: shape mismatch");
        Matrix<R> out(a.rows(), a.cols());
        for (std::size_t i = 0; i < out.rows(); ++i)
            for (std::size_t j = 0; j < out.cols(); ++j)
                out(i, j) = static_cast<R>(a(i, j)) - static_cast<R>(b(i, j));
        return out;
    }

    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator/(const Matrix<T>& a, const Matrix<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        if (a.rows() != b.rows() || a.cols() != b.cols())
            throw std::runtime_error("operator/: shape mismatch");
        Matrix<R> out(a.rows(), a.cols());
        for (std::size_t i = 0; i < out.rows(); ++i)
            for (std::size_t j = 0; j < out.cols(); ++j)
                out(i, j) = static_cast<R>(a(i, j)) / static_cast<R>(b(i, j));
        return out;
    }

    // Comparison elementwise -> Matrix<bool> (avoid CArray temporaries)
    template <typename T, typename U>
    Matrix<bool> operator<(const Matrix<T>& a, const Matrix<U>& b)
    {
        if (a.rows() != b.rows() || a.cols() != b.cols())
            throw std::runtime_error("operator<: shape mismatch");
        Matrix<bool> out(a.rows(), a.cols());
        for (std::size_t i = 0; i < out.rows(); ++i)
            for (std::size_t j = 0; j < out.cols(); ++j)
                out(i, j) = (static_cast<typename std::common_type<T, U>::type>(a(i, j)) < static_cast<typename std::common_type<T, U>::type>(b(i, j)));
        return out;
    }

    template <typename T, typename U>
    Matrix<bool> operator<=(const Matrix<T>& a, const Matrix<U>& b)
    {
        if (a.rows() != b.rows() || a.cols() != b.cols())
            throw std::runtime_error("operator<=: shape mismatch");
        Matrix<bool> out(a.rows(), a.cols());
        for (std::size_t i = 0; i < out.rows(); ++i)
            for (std::size_t j = 0; j < out.cols(); ++j)
                out(i, j) = (static_cast<typename std::common_type<T, U>::type>(a(i, j)) <= static_cast<typename std::common_type<T, U>::type>(b(i, j)));
        return out;
    }

    template <typename T, typename U>
    Matrix<bool> operator>(const Matrix<T>& a, const Matrix<U>& b)
    {
        if (a.rows() != b.rows() || a.cols() != b.cols())
            throw std::runtime_error("operator>: shape mismatch");
        Matrix<bool> out(a.rows(), a.cols());
        for (std::size_t i = 0; i < out.rows(); ++i)
            for (std::size_t j = 0; j < out.cols(); ++j)
                out(i, j) = (static_cast<typename std::common_type<T, U>::type>(a(i, j)) > static_cast<typename std::common_type<T, U>::type>(b(i, j)));
        return out;
    }

    template <typename T, typename U>
    Matrix<bool> operator>=(const Matrix<T>& a, const Matrix<U>& b)
    {
        if (a.rows() != b.rows() || a.cols() != b.cols())
            throw std::runtime_error("operator>=: shape mismatch");
        Matrix<bool> out(a.rows(), a.cols());
        for (std::size_t i = 0; i < out.rows(); ++i)
            for (std::size_t j = 0; j < out.cols(); ++j)
                out(i, j) = (static_cast<typename std::common_type<T, U>::type>(a(i, j)) >= static_cast<typename std::common_type<T, U>::type>(b(i, j)));
        return out;
    }

    // NOTE:
    // Matrix * Vector and Matrix * Matrix operator overloads are declared here but
    // implemented in linalg.h (after matmul definitions) to avoid a circular
    // instantiation/link error on MSVC. See linalg.h for implementations.
    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator*(const Matrix<T>& A, const Vector<U>& x);

    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator*(const Matrix<T>& A, const Matrix<U>& B);

    // Scalar ops for Matrix (scalar on right and left) with mixed scalar type
    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator+(const Matrix<T>& m, const U& scalar)
    {
        using R = typename std::common_type<T, U>::type;
        Matrix<R> r(m.rows(), m.cols());
        for (std::size_t i = 0; i < r.rows(); ++i)
            for (std::size_t j = 0; j < r.cols(); ++j)
                r(i, j) = static_cast<R>(m(i, j)) + static_cast<R>(scalar);
        return r;
    }

    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator+(const U& scalar, const Matrix<T>& m) { return m + scalar; }

    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator-(const Matrix<T>& m, const U& scalar)
    {
        using R = typename std::common_type<T, U>::type;
        Matrix<R> r(m.rows(), m.cols());
        for (std::size_t i = 0; i < r.rows(); ++i)
            for (std::size_t j = 0; j < r.cols(); ++j)
                r(i, j) = static_cast<R>(m(i, j)) - static_cast<R>(scalar);
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
        Matrix<R> r(m.rows(), m.cols());
        for (std::size_t i = 0; i < r.rows(); ++i)
            for (std::size_t j = 0; j < r.cols(); ++j)
                r(i, j) = static_cast<R>(m(i, j)) * static_cast<R>(scalar);
        return r;
    }

    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator*(const U& scalar, const Matrix<T>& m) { return m * scalar; }

    template <typename T, typename U>
    Matrix<typename std::common_type<T, U>::type> operator/(const Matrix<T>& m, const U& scalar)
    {
        using R = typename std::common_type<T, U>::type;
        Matrix<R> r(m.rows(), m.cols());
        for (std::size_t i = 0; i < r.rows(); ++i)
            for (std::size_t j = 0; j < r.cols(); ++j)
                r(i, j) = static_cast<R>(m(i, j)) / static_cast<R>(scalar);
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

    // simple pretty-print for debugging
    template <typename T>
    std::ostream& operator<<(std::ostream& os, const Matrix<T>& m)
    {
        for (std::size_t i = 0; i < m.rows(); ++i)
        {
            os << "[";
            for (std::size_t j = 0; j < m.cols(); ++j)
            {
                os << m(i, j);
                if (j + 1 < m.cols()) os << ", ";
            }
            os << "]";
            if (i + 1 < m.rows()) os << "\n";
        }
        return os;
    }

} // namespace CNum

#endif // NUMERICALS_MATRIX_H