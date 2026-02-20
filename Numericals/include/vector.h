#ifndef NUMERICALS_VECTOR_H
#define NUMERICALS_VECTOR_H

#include "CArray.h"
#include "ufuncs.h"
#include <initializer_list>
#include <type_traits>
#include <stdexcept>
#include <utility>
#include <numeric>
#include <cmath>

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

        // construct from underlying CArray (expects 1-D shape)
        explicit Vector(CArray<T>&& a)
            : arr_(std::move(a))
        {
            if (arr_.shape().size() != 1)
                throw std::runtime_error("Vector: underlying CArray must be 1-D");
        }

        std::size_t size() const noexcept { return arr_.size(); }
        bool empty() const noexcept { return size() == 0; }

        // data() left as-is; callers should avoid using data() for CArray<bool>
        T* data() noexcept { return arr_.data(); }
        const T* data() const noexcept { return arr_.data(); }

        // iterator support (pointer-based; works even if CArray doesn't expose iterators)
        value_type* begin() noexcept { return data(); }
        value_type* end() noexcept { return data() + size(); }
        const value_type* begin() const noexcept { return data(); }
        const value_type* end() const noexcept { return data() + size(); }
        const value_type* cbegin() const noexcept { return data(); }
        const value_type* cend() const noexcept { return data() + size(); }

        // front / back helpers
        value_type& front()
        {
            if (empty()) throw std::out_of_range("Vector::front on empty");
            return arr_[0];
        }
        const value_type& front() const
        {
            if (empty()) throw std::out_of_range("Vector::front on empty");
            return arr_[0];
        }
        value_type& back()
        {
            if (empty()) throw std::out_of_range("Vector::back on empty");
            return arr_[size() - 1];
        }
        const value_type& back() const
        {
            if (empty()) throw std::out_of_range("Vector::back on empty");
            return arr_[size() - 1];
        }

        // unchecked element access (fast)
        auto operator[](std::size_t i) -> decltype(std::declval<CArray<T>&>()[i])
        {
            return arr_[i];
        }
        auto operator[](std::size_t i) const -> decltype(std::declval<const CArray<T>&>()[i])
        {
            return arr_[i];
        }

        // bounds-checked access (throws std::out_of_range)
        auto at(std::size_t i) -> decltype(std::declval<CArray<T>&>().at(shape_t{}))
        {
            if (i >= size())
                throw std::out_of_range("Vector::at: index out of bounds");
            // CArray::operator[] is efficient; use it to preserve proxy/reference semantics.
            return arr_[i];
        }

        auto at(std::size_t i) const -> decltype(std::declval<const CArray<T>&>().at(shape_t{}))
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

        // sum of elements
        auto sum() const -> typename std::common_type<T>::type
        {
            using R = typename std::common_type<T>::type;
            R s = R();
            for (std::size_t i = 0; i < size(); ++i) s += static_cast<R>((*this)[i]);
            return s;
        }

        // Euclidean norm (returns floating type)
        auto norm() const -> typename std::common_type<T, long double>::type
        {
            using R = typename std::common_type<T, long double>::type;
            R s = R();
            for (std::size_t i = 0; i < size(); ++i)
                s += static_cast<R>((*this)[i]) * static_cast<R>((*this)[i]);
            return static_cast<typename std::common_type<T, long double>::type>(std::sqrt(static_cast<long double>(s)));
        }

        // return underlying CArray (copy)
        CArray<T> to_carray() const { return arr_; }

        // value equality operators (compare full vector)
        bool operator==(const Vector<T>& other) const noexcept
        {
            if (size() != other.size()) return false;
            for (std::size_t i = 0; i < size(); ++i)
            {
                if (!((*this)[i] == other[i])) return false;
            }
            return true;
        }

        bool operator!=(const Vector<T>& other) const noexcept { return !(*this == other); }

    private:
        CArray<T> arr_;
    };

    // unary negation
    template <typename T>
    Vector<T> operator-(const Vector<T>& v)
    {
        Vector<T> r(carray_cast<T>(v.to_carray()));
        for (std::size_t i = 0; i < r.size(); ++i) r[i] = -r[i];
        return r;
    }

    // elementwise in-place compound with another Vector (simple, checks size)
    template <typename T, typename U>
    Vector<T>& operator+=(Vector<T>& v, const Vector<U>& other)
    {
        if (v.size() != other.size())
            throw std::runtime_error("Vector::operator+= size mismatch");
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = v[i] + static_cast<T>(other[i]);
        return v;
    }

    template <typename T, typename U>
    Vector<T>& operator-=(Vector<T>& v, const Vector<U>& other)
    {
        if (v.size() != other.size())
            throw std::runtime_error("Vector::operator-= size mismatch");
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = v[i] - static_cast<T>(other[i]);
        return v;
    }

    template <typename T, typename U>
    Vector<T>& operator*=(Vector<T>& v, const Vector<U>& other)
    {
        if (v.size() != other.size())
            throw std::runtime_error("Vector::operator*= size mismatch");
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = v[i] * static_cast<T>(other[i]);
        return v;
    }

    template <typename T, typename U>
    Vector<T>& operator/=(Vector<T>& v, const Vector<U>& other)
    {
        if (v.size() != other.size())
            throw std::runtime_error("Vector::operator/= size mismatch");
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = v[i] / static_cast<T>(other[i]);
        return v;
    }

    // elementwise Vector ops use ufuncs (require same shape or broadcasting)
    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator+(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a.to_carray());
        auto nb = carray_cast<R>(b.to_carray());
        return Vector<R>(CNum::add(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator-(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a.to_carray());
        auto nb = carray_cast<R>(b.to_carray());
        return Vector<R>(CNum::sub(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator*(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a.to_carray());
        auto nb = carray_cast<R>(b.to_carray());
        return Vector<R>(CNum::mul(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator/(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a.to_carray());
        auto nb = carray_cast<R>(b.to_carray());
        return Vector<R>(CNum::div(std::move(na), std::move(nb)));
    }

    // Bitwise elementwise operators (integers or types supporting bit ops)
    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator&(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a.to_carray());
        auto nb = carray_cast<R>(b.to_carray());
        return Vector<R>(CNum::bit_and(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator|(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a.to_carray());
        auto nb = carray_cast<R>(b.to_carray());
        return Vector<R>(CNum::bit_or(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator^(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a.to_carray());
        auto nb = carray_cast<R>(b.to_carray());
        return Vector<R>(CNum::bit_xor(std::move(na), std::move(nb)));
    }

    // Comparison elementwise -> Vector<bool>
    // Note: the elementwise equality/inequality operators that previously
    // conflicted with the member value-comparison operators were removed.
    template <typename T, typename U>
    Vector<bool> operator<(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a.to_carray());
        auto nb = carray_cast<R>(b.to_carray());
        return Vector<bool>(CNum::lt(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Vector<bool> operator<=(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a.to_carray());
        auto nb = carray_cast<R>(b.to_carray());
        return Vector<bool>(CNum::le(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Vector<bool> operator>(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a.to_carray());
        auto nb = carray_cast<R>(b.to_carray());
        return Vector<bool>(CNum::gt(std::move(na), std::move(nb)));
    }

    template <typename T, typename U>
    Vector<bool> operator>=(const Vector<T>& a, const Vector<U>& b)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a.to_carray());
        auto nb = carray_cast<R>(b.to_carray());
        return Vector<bool>(CNum::ge(std::move(na), std::move(nb)));
    }

    // Broadcasting between Vector (1-D) and CArray (N-D).
    // These return CArray<R> because result can be higher-rank after broadcasting.

    // Vector op CArray
    template <typename T, typename U>
    CArray<typename std::common_type<T, U>::type> operator+(const Vector<T>& v, const CArray<U>& a)
    {
        using R = typename std::common_type<T, U>::type;
        auto nv = carray_cast<R>(v.to_carray());
        auto na = carray_cast<R>(a);
        return CNum::add(std::move(nv), std::move(na));
    }

    template <typename T, typename U>
    CArray<typename std::common_type<T, U>::type> operator-(const Vector<T>& v, const CArray<U>& a)
    {
        using R = typename std::common_type<T, U>::type;
        auto nv = carray_cast<R>(v.to_carray());
        auto na = carray_cast<R>(a);
        return CNum::sub(std::move(nv), std::move(na));
    }

    template <typename T, typename U>
    CArray<typename std::common_type<T, U>::type> operator*(const Vector<T>& v, const CArray<U>& a)
    {
        using R = typename std::common_type<T, U>::type;
        auto nv = carray_cast<R>(v.to_carray());
        auto na = carray_cast<R>(a);
        return CNum::mul(std::move(nv), std::move(na));
    }

    template <typename T, typename U>
    CArray<typename std::common_type<T, U>::type> operator/(const Vector<T>& v, const CArray<U>& a)
    {
        using R = typename std::common_type<T, U>::type;
        auto nv = carray_cast<R>(v.to_carray());
        auto na = carray_cast<R>(a);
        return CNum::div(std::move(nv), std::move(na));
    }

    template <typename T, typename U>
    CArray<bool> operator<(const Vector<T>& v, const CArray<U>& a)
    {
        using R = typename std::common_type<T, U>::type;
        auto nv = carray_cast<R>(v.to_carray());
        auto na = carray_cast<R>(a);
        return CNum::lt(std::move(nv), std::move(na));
    }

    template <typename T, typename U>
    CArray<bool> operator<=(const Vector<T>& v, const CArray<U>& a)
    {
        using R = typename std::common_type<T, U>::type;
        auto nv = carray_cast<R>(v.to_carray());
        auto na = carray_cast<R>(a);
        return CNum::le(std::move(nv), std::move(na));
    }

    template <typename T, typename U>
    CArray<bool> operator>(const Vector<T>& v, const CArray<U>& a)
    {
        using R = typename std::common_type<T, U>::type;
        auto nv = carray_cast<R>(v.to_carray());
        auto na = carray_cast<R>(a);
        return CNum::gt(std::move(nv), std::move(na));
    }

    template <typename T, typename U>
    CArray<bool> operator>=(const Vector<T>& v, const CArray<U>& a)
    {
        using R = typename std::common_type<T, U>::type;
        auto nv = carray_cast<R>(v.to_carray());
        auto na = carray_cast<R>(a);
        return CNum::ge(std::move(nv), std::move(na));
    }

    // CArray op Vector (reverse order)
    template <typename T, typename U>
    CArray<typename std::common_type<T, U>::type> operator+(const CArray<T>& a, const Vector<U>& v)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a);
        auto nv = carray_cast<R>(v.to_carray());
        return CNum::add(std::move(na), std::move(nv));
    }

    template <typename T, typename U>
    CArray<typename std::common_type<T, U>::type> operator-(const CArray<T>& a, const Vector<U>& v)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a);
        auto nv = carray_cast<R>(v.to_carray());
        return CNum::sub(std::move(na), std::move(nv));
    }

    template <typename T, typename U>
    CArray<typename std::common_type<T, U>::type> operator*(const CArray<T>& a, const Vector<U>& v)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a);
        auto nv = carray_cast<R>(v.to_carray());
        return CNum::mul(std::move(na), std::move(nv));
    }

    template <typename T, typename U>
    CArray<typename std::common_type<T, U>::type> operator/(const CArray<T>& a, const Vector<U>& v)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a);
        auto nv = carray_cast<R>(v.to_carray());
        return CNum::div(std::move(na), std::move(nv));
    }

    template <typename T, typename U>
    CArray<bool> operator<(const CArray<T>& a, const Vector<U>& v)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a);
        auto nv = carray_cast<R>(v.to_carray());
        return CNum::lt(std::move(na), std::move(nv));
    }

    template <typename T, typename U>
    CArray<bool> operator<=(const CArray<T>& a, const Vector<U>& v)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a);
        auto nv = carray_cast<R>(v.to_carray());
        return CNum::le(std::move(na), std::move(nv));
    }

    template <typename T, typename U>
    CArray<bool> operator>(const CArray<T>& a, const Vector<U>& v)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a);
        auto nv = carray_cast<R>(v.to_carray());
        return CNum::gt(std::move(na), std::move(nv));
    }

    template <typename T, typename U>
    CArray<bool> operator>=(const CArray<T>& a, const Vector<U>& v)
    {
        using R = typename std::common_type<T, U>::type;
        auto na = carray_cast<R>(a);
        auto nv = carray_cast<R>(v.to_carray());
        return CNum::ge(std::move(na), std::move(nv));
    }

    // Scalar ops (vector <op> scalar) and (scalar <op> vector) with mixed scalar type
    template <typename T, typename U>
    Vector<typename std::common_type<T, U>::type> operator+(const Vector<T>& v, const U& scalar)
    {
        using R = typename std::common_type<T, U>::type;
        Vector<R> r(carray_cast<R>(v.to_carray()));
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
        Vector<R> r(carray_cast<R>(v.to_carray()));
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
        Vector<R> r(carray_cast<R>(v.to_carray()));
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
        Vector<R> r(carray_cast<R>(v.to_carray()));
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