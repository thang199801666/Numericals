#ifndef NUMERICALS_CARRAY_H
#define NUMERICALS_CARRAY_H

#include "dtypes.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <type_traits>

namespace CNum
{

    // Dense N-dimensional array, row-major storage.
    // Renamed from 'NdArray' to 'CArray'.
    template <typename T>
    class CArray
    {
    public:
        using value_type = T;

        CArray() = default;

        explicit CArray(const shape_t& shape)
            : shape_(shape)
            , strides_(compute_strides(shape_))
            , data_(compute_size(shape_))
        {
        }

        CArray(const shape_t& shape, const T& fill_value)
            : shape_(shape)
            , strides_(compute_strides(shape_))
            , data_(compute_size(shape_), fill_value)
        {
        }

        // Construct from initializer list for 1-D arrays
        CArray(std::initializer_list<T> list)
            : shape_{ list.size() }
            , strides_(compute_strides(shape_))
            , data_(list.begin(), list.end())
        {
        }

        std::size_t ndim() const noexcept { return shape_.size(); }
        const shape_t& shape() const noexcept { return shape_; }
        const strides_t& strides() const noexcept { return strides_; }
        std::size_t size() const noexcept { return data_.size(); }

        // data() remains available for types that provide .data()
        // For std::vector<bool> .data() is not available; caller must avoid calling data() for bool arrays.
        T* data() noexcept { return data_.data(); }
        const T* data() const noexcept { return data_.data(); }

        // Linear index access
        // Use std::vector<T>::reference / const_reference so proxy references (e.g. vector<bool>) work.
        auto operator[](std::size_t idx) -> typename std::vector<T>::reference
        {
            return data_[idx];
        }
        auto operator[](std::size_t idx) const -> typename std::vector<T>::const_reference
        {
            return data_[idx];
        }

        // Multidimensional access with bounds checking
        auto at(const shape_t& indices) -> typename std::vector<T>::reference
        {
            return data_.at(offset(indices));
        }

        auto at(const shape_t& indices) const -> typename std::vector<T>::const_reference
        {
            return data_.at(offset(indices));
        }

        // Convenience fixed-arity operator() for small ranks (variadic)
        template <typename... Idx>
        decltype(auto) operator()(Idx... idxs)
        {
            static_assert(sizeof...(Idx) > 0, "Must provide at least one index");
            shape_t indices = { static_cast<std::size_t>(idxs)... };
            return at(indices);
        }

        template <typename... Idx>
        decltype(auto) operator()(Idx... idxs) const
        {
            static_assert(sizeof...(Idx) > 0, "Must provide at least one index");
            shape_t indices = { static_cast<std::size_t>(idxs)... };
            return at(indices);
        }

        void reshape(const shape_t& new_shape)
        {
            std::size_t new_size = compute_size(new_shape);
            if (new_size != size())
            {
                throw std::runtime_error("reshape: total size must remain unchanged");
            }
            shape_ = new_shape;
            strides_ = compute_strides(shape_);
        }

        void fill(const T& value)
        {
            std::fill(data_.begin(), data_.end(), value);
        }

    private:
        shape_t shape_;
        strides_t strides_;
        std::vector<T> data_;

        static strides_t compute_strides(const shape_t& shape)
        {
            strides_t s(shape.size());
            if (shape.empty())
                return s;
            std::size_t running = 1;
            for (std::size_t i = shape.size(); i-- > 0;)
            {
                s[i] = running;
                running *= shape[i];
            }
            return s;
        }

        static std::size_t compute_size(const shape_t& shape)
        {
            if (shape.empty())
                return 0;
            return std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1), std::multiplies<std::size_t>());
        }

        std::size_t offset(const shape_t& indices) const
        {
            if (indices.size() != shape_.size())
                throw std::out_of_range("index rank mismatch");
            std::size_t off = 0;
            for (std::size_t i = 0; i < shape_.size(); ++i)
            {
                if (indices[i] >= shape_[i])
                    throw std::out_of_range("index out of bounds");
                off += indices[i] * strides_[i];
            }
            return off;
        }
    };

    // Generic CArray cast helper: produce CArray<R> from CArray<S> by elementwise static_cast.
    template <typename R, typename S>
    inline CArray<R> carray_cast(const CArray<S>& src)
    {
        CArray<R> out(src.shape());
        for (std::size_t i = 0; i < src.size(); ++i)
        {
            out[i] = static_cast<R>(src[i]);
        }
        return out;
    }

} // namespace CNum

#endif // NUMERICALS_CARRAY_H