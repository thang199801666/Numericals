#ifndef NUMERICALS_VECTOR_H
#define NUMERICALS_VECTOR_H

#include "ndarray.h"
#include "ufuncs.h"
#include <initializer_list>
#include <type_traits>
#include <stdexcept>
#include <utility>

namespace CNum
{

    template <typename T>
    class Vector
    {
    public:
        using value_type = T;

        Vector() = default;

        explicit Vector(std::size_t n)
            : arr_(shape_t{ n })
        {
        }

        Vector(std::size_t n, const T& fill_value)
            : arr_(shape_t{ n }, fill_value)
        {
        }

        Vector(std::initializer_list<T> list)
            : arr_(list)
        {
        }

        // construct from underlying NdArray (expects 1-D shape)
        explicit Vector(NdArray<T>&& a)
            : arr_(std::move(a))
        {
            if (arr_.shape().size() != 1)
                throw std::runtime_error("Vector: underlying NdArray must be 1-D");
        }

        std::size_t size() const noexcept { return arr_.size(); }

        // data() left as-is; callers should avoid using data() for NdArray<bool>
        T* data() noexcept { return arr_.data(); }
        const T* data() const noexcept { return arr_.data(); }

        // unchecked element access (fast)
        auto operator[](std::size_t i) -> decltype(std::declval<NdArray<T>&>()[i])
        {
            return arr_[i];
        }
        auto operator[](std::size_t i) const -> decltype(std::declval<const NdArray<T>&>()[i])
        {
            return arr_[i];
        }

        // bounds-checked access (throws std::out_of_range)
        auto at(std::size_t i) -> decltype(std::declval<NdArray<T>&>().at(shape_t{}))
        {
            if (i >= size())
                throw std::out_of_range("Vector::at: index out of bounds");
            // NdArray::operator[] is efficient; use it to preserve proxy/reference semantics.
            return arr_[i];
        }

        auto at(std::size_t i) const -> decltype(std::declval<const NdArray<T>&>().at(shape_t{}))
        {
            if (i >= size())
                throw std::out_of_range("Vector::at: index out of bounds");
            return arr_[i];
        }

        // dot product (member)
        template <typename U>
        auto dot(const Vector<U>& other) const -> typename std::common_type<T, U>::type
        {
            using R = typename std::common_type<T, U>::type;
            if (size() != other.size())
                throw std::runtime_error("Vector::dot: size mismatch");
            R sum = R();
            for (std::size_t i = 0; i < size(); ++i)
                sum += static_cast<R>((*this)[i]) * static_cast<R>(other[i]);
            return sum;
        }

        // cross product (member) — only for 3D vectors
        template <typename U>
        Vector<typename std::common_type<T, U>::type> cross(const Vector<U>& other) const
        {
            using R = typename std::common_type<T, U>::type;
            if (size() != 3 || other.size() != 3)
                throw std::runtime_error("Vector::cross requires 3D vectors");
            Vector<R> r(3);
            r[0] = static_cast<R>((*this)[1]) * static_cast<R>(other[2]) - static_cast<R>((*this)[2]) * static_cast<R>(other[1]);
            r[1] = static_cast<R>((*this)[2]) * static_cast<R>(other[0]) - static_cast<R>((*this)[0]) * static_cast<R>(other[2]);
            r[2] = static_cast<R>((*this)[0]) * static_cast<R>(other[1]) - static_cast<R>((*this)[1]) * static_cast<R>(other[0]);
            return r;
        }

        // return underlying NdArray (copy)
        NdArray<T> to_ndarray() const { return arr_; }

    private:
        NdArray<T> arr_;
    };

    // elementwise Vector ops use ufuncs (require same shape)
    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator+(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = ndarray_cast<R>(a.to_ndarray());
        auto nb = ndarray_cast<R>(b.to_ndarray());
        return Vector<R>(CNum::add(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator-(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = ndarray_cast<R>(a.to_ndarray());
        auto nb = ndarray_cast<R>(b.to_ndarray());
        return Vector<R>(CNum::sub(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator*(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = ndarray_cast<R>(a.to_ndarray());
        auto nb = ndarray_cast<R>(b.to_ndarray());
        return Vector<R>(CNum::mul(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator/(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = ndarray_cast<R>(a.to_ndarray());
        auto nb = ndarray_cast<R>(b.to_ndarray());
        return Vector<R>(CNum::div(std::move(na), std::move(nb)));
    }

    // Bitwise elementwise operators (integers or types supporting bit ops)
    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator&(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = ndarray_cast<R>(a.to_ndarray());
        auto nb = ndarray_cast<R>(b.to_ndarray());
        return Vector<R>(CNum::bit_and(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator|(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = ndarray_cast<R>(a.to_ndarray());
        auto nb = ndarray_cast<R>(b.to_ndarray());
        return Vector<R>(CNum::bit_or(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator^(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = ndarray_cast<R>(a.to_ndarray());
        auto nb = ndarray_cast<R>(b.to_ndarray());
        return Vector<R>(CNum::bit_xor(std::move(na), std::move(nb)));
    }

    // Comparison elementwise -> Vector<bool>
    template <typename T, typename U>
    Vector<bool> operator==(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = ndarray_cast<R>(a.to_ndarray());
        auto nb = ndarray_cast<R>(b.to_ndarray());
        return Vector<bool>(CNum::eq(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Vector<bool> operator!=(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = ndarray_cast<R>(a.to_ndarray());
        auto nb = ndarray_cast<R>(b.to_ndarray());
        return Vector<bool>(CNum::neq(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Vector<bool> operator<(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = ndarray_cast<R>(a.to_ndarray());
        auto nb = ndarray_cast<R>(b.to_ndarray());
        return Vector<bool>(CNum::lt(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Vector<bool> operator<=(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = ndarray_cast<R>(a.to_ndarray());
        auto nb = ndarray_cast<R>(b.to_ndarray());
        return Vector<bool>(CNum::le(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Vector<bool> operator>(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = ndarray_cast<R>(a.to_ndarray());
        auto nb = ndarray_cast<R>(b.to_ndarray());
        return Vector<bool>(CNum::gt(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Vector<bool> operator>=(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = ndarray_cast<R>(a.to_ndarray());
        auto nb = ndarray_cast<R>(b.to_ndarray());
        return Vector<bool>(CNum::ge(std::move(na), std::move(nb)));
    }

    // Scalar ops (vector <op> scalar) and (scalar <op> vector) with mixed scalar type
    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator+(const Vector<T>& v, const U& scalar)
    {
        using R = typename std::common_type<T, U>::type;
        Vector<R> r(ndarray_cast<R>(v.to_ndarray()));
        for (std::size_t i = 0; i < r.size(); ++i) r[i] = r[i] + static_cast<R>(scalar);
        return r;
    }

    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator+(const U& scalar, const Vector<T>& v)
    {
        return v + scalar;
    }

    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator-(const Vector<T>& v, const U& scalar)
    {
        using R = typename std::common_type<T, U>::type;
        Vector<R> r(ndarray_cast<R>(v.to_ndarray()));
        for (std::size_t i = 0; i < r.size(); ++i) r[i] = r[i] - static_cast<R>(scalar);
        return r;
    }

    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator-(const U& scalar, const Vector<T>& v)
    {
        using R = typename std::common_type<T, U>::type;
        Vector<R> r(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) r[i] = static_cast<R>(scalar) - static_cast<R>(v[i]);
        return r;
    }

    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator*(const Vector<T>& v, const U& scalar)
    {
        using R = typename std::common_type<T, U>::type;
        Vector<R> r(ndarray_cast<R>(v.to_ndarray()));
        for (std::size_t i = 0; i < r.size(); ++i) r[i] = r[i] * static_cast<R>(scalar);
        return r;
    }

    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator*(const U& scalar, const Vector<T>& v)
    {
        return v * scalar;
    }

    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator/(const Vector<T>& v, const U& scalar)
    {
        using R = typename std::common_type<T, U>::type;
        Vector<R> r(ndarray_cast<R>(v.to_ndarray()));
        for (std::size_t i = 0; i < r.size(); ++i) r[i] = r[i] / static_cast<R>(scalar);
        return r;
    }

    // Compound assignment with scalar (in-place) - only when scalar converts to value_type
    template <typename T>
    Vector<T>& operator+=(Vector<T>& v, const T& scalar)
    {
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = v[i] + scalar;
        return v;
    }

    template <typename T>
    Vector<T>& operator-=(Vector<T>& v, const T& scalar)
    {
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = v[i] - scalar;
        return v;
    }

    template <typename T>
    Vector<T>& operator*=(Vector<T>& v, const T& scalar)
    {
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = v[i] * scalar;
        return v;
    }

    template <typename T>
    Vector<T>& operator/=(Vector<T>& v, const T& scalar)
    {
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = v[i] / scalar;
        return v;
    }

} // namespace CNum

#endif // NUMERICALS_VECTOR_H