#ifndef NUMERICALS_VIEWS_H
#define NUMERICALS_VIEWS_H

#include "dtypes.h"
#include "ndarray.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace numericals
{

    // Small helper: convert linear index to multidim indices (row-major)
    inline void linear_to_indices(std::size_t linear, const shape_t& shape, shape_t& out_indices)
    {
        const std::size_t n = shape.size();
        out_indices.assign(n, 0);
        if (n == 0) return;
        // compute strides for row-major
        std::size_t denom = 1;
        for (std::size_t i = 1; i < n; ++i) denom *= shape[i];
        std::size_t rem = linear;
        for (std::size_t i = 0; i < n; ++i)
        {
            std::size_t stride = denom;
            out_indices[i] = (stride > 0) ? (rem / stride) : 0;
            if (stride > 0) rem = rem % stride;
            if (i + 1 < n) denom /= shape[i + 1];
        }
    }

    // Non-owning view over dense n-d array (row-major base strides allowed)
    template <typename T>
    class NdView
    {
    public:
        using value_type = T;

        NdView() = default;

        // from raw pointer, shape and strides (strides in elements)
        NdView(T* data, const shape_t& shape, const strides_t& strides)
            : data_(data), shape_(shape), strides_(strides)
        {
            if (shape_.size() != strides_.size())
                throw std::runtime_error("NdView: shape/strides size mismatch");
        }

        // from NdArray (non-owning)
        explicit NdView(NdArray<T>& arr)
            : data_(arr.data()), shape_(arr.shape()), strides_(arr.strides())
        {
        }

        std::size_t ndim() const noexcept { return shape_.size(); }
        const shape_t& shape() const noexcept { return shape_; }
        const strides_t& strides() const noexcept { return strides_; }
        std::size_t size() const noexcept
        {
            if (shape_.empty()) return 0;
            std::size_t s = 1;
            for (auto v : shape_) s *= v;
            return s;
        }
        T* data() noexcept { return data_; }
        const T* data() const noexcept { return data_; }

        // linear index access (maps into possibly non-contiguous data via strides)
        T& operator[](std::size_t linear)
        {
            shape_t indices;
            linear_to_indices(linear, shape_, indices);
            return at(indices);
        }

        const T& operator[](std::size_t linear) const
        {
            shape_t indices;
            linear_to_indices(linear, shape_, indices);
            return at(indices);
        }

        // access by multidim indices
        T& at(const shape_t& indices)
        {
            if (indices.size() != shape_.size()) throw std::out_of_range("NdView::at rank mismatch");
            std::size_t off = 0;
            for (std::size_t i = 0; i < shape_.size(); ++i)
            {
                if (indices[i] >= shape_[i]) throw std::out_of_range("NdView::at index out of bounds");
                off += indices[i] * strides_[i];
            }
            return data_[off];
        }

        const T& at(const shape_t& indices) const
        {
            if (indices.size() != shape_.size()) throw std::out_of_range("NdView::at rank mismatch");
            std::size_t off = 0;
            for (std::size_t i = 0; i < shape_.size(); ++i)
            {
                if (indices[i] >= shape_[i]) throw std::out_of_range("NdView::at index out of bounds");
                off += indices[i] * strides_[i];
            }
            return data_[off];
        }

        // simple flat iterator that iterates by linear index and uses operator[]
        class FlatIterator
        {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = T;
            using difference_type = std::ptrdiff_t;
            using pointer = T*;
            using reference = T&;

            FlatIterator(NdView<T>* view, std::size_t pos) : view_(view), pos_(pos) {}
            reference operator*() const { return (*view_)[pos_]; }
            pointer operator->() const { return &(*view_)[pos_]; }
            FlatIterator& operator++() { ++pos_; return *this; }
            FlatIterator operator++(int) { FlatIterator tmp = *this; ++(*this); return tmp; }
            bool operator==(const FlatIterator& o) const { return pos_ == o.pos_ && view_ == o.view_; }
            bool operator!=(const FlatIterator& o) const { return !(*this == o); }

        private:
            NdView<T>* view_;
            std::size_t pos_;
        };

        FlatIterator begin() { return FlatIterator(this, 0); }
        FlatIterator end() { return FlatIterator(this, size()); }
        FlatIterator begin() const { return FlatIterator(const_cast<NdView<T>*>(this), 0); }
        FlatIterator end() const { return FlatIterator(const_cast<NdView<T>*>(this), size()); }

    private:
        T* data_ = nullptr;
        shape_t shape_;
        strides_t strides_;
    };

} // namespace numericals

#endif // NUMERICALS_VIEWS_H