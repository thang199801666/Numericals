
#ifndef NUMERICALS_UFUNCS_H
#define NUMERICALS_UFUNCS_H

#include "ndarray.h"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>

namespace CNum
{

    // Helper: pretty-print shape
    inline std::string shape_to_string(const shape_t& s)
    {
        std::ostringstream oss;
        oss << "(";
        for (std::size_t i = 0; i < s.size(); ++i)
        {
            if (i) oss << ", ";
            oss << s[i];
        }
        oss << ")";
        return oss.str();
    }

    // Convert indices vector to printable string
    inline std::string indices_to_string(const shape_t& ind)
    {
        std::ostringstream oss;
        oss << "(";
        for (std::size_t i = 0; i < ind.size(); ++i)
        {
            if (i) oss << ", ";
            oss << ind[i];
        }
        oss << ")";
        return oss.str();
    }

    // Compute broadcasted shape of two operands. Throws on incompatible shapes.
    inline shape_t broadcast_shape(const shape_t& a, const shape_t& b)
    {
        const std::size_t na = a.size();
        const std::size_t nb = b.size();
        const std::size_t n = std::max(na, nb);
        shape_t result(n, 1);

        // Align and compare from the right (least-significant) dimensions.
        for (std::size_t i = 0; i < n; ++i)
        {
            // position from the right: 0 is last dim
            std::size_t ia = (i < na) ? a[na - 1 - i] : 1;
            std::size_t ib = (i < nb) ? b[nb - 1 - i] : 1;

            if (ia != ib && ia != 1 && ib != 1)
            {
                std::ostringstream msg;
                msg << "broadcast_shape: incompatible shapes for broadcasting: "
                    << shape_to_string(a) << " vs " << shape_to_string(b)
                    << " (mismatch at right-offset " << i << ": " << ia << " vs " << ib << ")";
                throw std::runtime_error(msg.str());
            }

            // store into result aligned from the right
            result[n - 1 - i] = std::max(ia, ib);
        }

        return result;
    }

    // Compute strides for a shape (row-major)
    inline strides_t compute_strides_for_shape(const shape_t& shape)
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

    // Convert a flat linear index into multidimensional indices for a given shape (row-major)
    inline void linear_to_indices(std::size_t linear, const shape_t& shape, shape_t& out_indices)
    {
        std::size_t n = shape.size();
        out_indices.assign(n, 0);
        if (n == 0)
            return;
        strides_t s = compute_strides_for_shape(shape);
        for (std::size_t i = 0; i < n; ++i)
        {
            out_indices[i] = (s[i] == 0) ? 0 : (linear / s[i]) % shape[i];
        }
    }

    // Map broadcasted indices to offset in original array
    inline std::size_t indices_to_offset_with_broadcast(const shape_t& bshape,
        const shape_t& orig_shape,
        const strides_t& orig_strides,
        const shape_t& indices)
    {
        // bshape and indices are same length; orig_shape and orig_strides are shorter or equal.
        const std::size_t nd = bshape.size();
        const std::size_t od = orig_shape.size();

        if (indices.size() != nd)
            throw std::runtime_error("indices_to_offset_with_broadcast: indices length mismatch with broadcast shape");

        // Align original shape/strides to the right of broadcast shape.
        const std::size_t right_gap = (nd > od) ? (nd - od) : 0;
        std::size_t offset = 0;

        // For each original dimension j (0..od-1), find the corresponding index in 'indices'.
        for (std::size_t j = 0; j < od; ++j)
        {
            std::size_t idx_in_bshape = j + right_gap; // position in the broadcasted indices
            if (idx_in_bshape >= indices.size())
            {
                std::ostringstream msg;
                msg << "indices_to_offset_with_broadcast: computed idx_in_bshape out of range. "
                    << "bshape=" << shape_to_string(bshape) << ", orig_shape=" << shape_to_string(orig_shape)
                    << ", indices=" << indices_to_string(indices) << ", idx_in_bshape=" << idx_in_bshape;
                throw std::runtime_error(msg.str());
            }

            std::size_t orig_dim = orig_shape[j];
            std::size_t orig_stride = (j < orig_strides.size()) ? orig_strides[j] : 0;

            std::size_t idx = indices[idx_in_bshape];
            std::size_t orig_idx = (orig_dim == 1) ? 0 : idx;

            offset += orig_idx * orig_stride;
        }

        return offset;
    }

    // Generic elementwise binary op with broadcasting (with bounds checks and diagnostics)
    template <typename T, typename BinaryOp>
    NdArray<T> elementwise_binary_op(const NdArray<T>& a, const NdArray<T>& b, BinaryOp op)
    {
        shape_t out_shape = broadcast_shape(a.shape(), b.shape());
        NdArray<T> out(out_shape);

        std::size_t total = out.size();
        shape_t indices;
        for (std::size_t linear = 0; linear < total; ++linear)
        {
            linear_to_indices(linear, out_shape, indices);

            std::size_t off_a = indices_to_offset_with_broadcast(out_shape, a.shape(), a.strides(), indices);
            std::size_t off_b = indices_to_offset_with_broadcast(out_shape, b.shape(), b.strides(), indices);

            // defensive checks before raw data access
            if (off_a >= a.size() || off_b >= b.size())
            {
                std::ostringstream msg;
                msg << "elementwise_binary_op: computed offset out of range.\n"
                    << "a.shape=" << shape_to_string(a.shape()) << ", a.size=" << a.size() << ", off_a=" << off_a << "\n"
                    << "b.shape=" << shape_to_string(b.shape()) << ", b.size=" << b.size() << ", off_b=" << off_b << "\n"
                    << "out_shape=" << shape_to_string(out_shape) << ", linear_index=" << linear
                    << ", indices=" << indices_to_string(indices);
                throw std::runtime_error(msg.str());
            }

            out[linear] = op(a.data()[off_a], b.data()[off_b]);
        }

        return out;
    }

    // Convenience ufuncs (arithmetic)
    template <typename T>
    NdArray<T> add(const NdArray<T>& a, const NdArray<T>& b)
    {
        return elementwise_binary_op(a, b, std::plus<T>{});
    }

    template <typename T>
    NdArray<T> sub(const NdArray<T>& a, const NdArray<T>& b)
    {
        return elementwise_binary_op(a, b, std::minus<T>{});
    }

    template <typename T>
    NdArray<T> mul(const NdArray<T>& a, const NdArray<T>& b)
    {
        return elementwise_binary_op(a, b, std::multiplies<T>{});
    }

    template <typename T>
    NdArray<T> div(const NdArray<T>& a, const NdArray<T>& b)
    {
        return elementwise_binary_op(a, b, std::divides<T>{});
    }

    // Bitwise (elementwise) — mixed-type support by casting to common_type
    template <typename A, typename B>
    inline NdArray<typename std::common_type<A, B>::type> bit_and(const NdArray<A>& a, const NdArray<B>& b)
    {
        using R = typename std::common_type<A, B>::type;
        auto na = ndarray_cast<R>(a);
        auto nb = ndarray_cast<R>(b);
        return elementwise_binary_op<R>(na, nb, std::bit_and<R>{});
    }

    template <typename A, typename B>
    inline NdArray<typename std::common_type<A, B>::type> bit_or(const NdArray<A>& a, const NdArray<B>& b)
    {
        using R = typename std::common_type<A, B>::type;
        auto na = ndarray_cast<R>(a);
        auto nb = ndarray_cast<R>(b);
        return elementwise_binary_op<R>(na, nb, std::bit_or<R>{});
    }

    template <typename A, typename B>
    inline NdArray<typename std::common_type<A, B>::type> bit_xor(const NdArray<A>& a, const NdArray<B>& b)
    {
        using R = typename std::common_type<A, B>::type;
        auto na = ndarray_cast<R>(a);
        auto nb = ndarray_cast<R>(b);
        return elementwise_binary_op<R>(na, nb, std::bit_xor<R>{});
    }

    // Comparison ops (elementwise) produce NdArray<bool>
    template <typename A, typename B>
    inline NdArray<bool> eq(const NdArray<A>& a, const NdArray<B>& b)
    {
        using R = typename std::common_type<A, B>::type;
        auto na = ndarray_cast<R>(a);
        auto nb = ndarray_cast<R>(b);
        shape_t out_shape = broadcast_shape(na.shape(), nb.shape());
        NdArray<bool> out(out_shape);
        std::size_t total = out.size();
        shape_t indices;
        for (std::size_t linear = 0; linear < total; ++linear)
        {
            linear_to_indices(linear, out_shape, indices);
            std::size_t off_a = indices_to_offset_with_broadcast(out_shape, na.shape(), na.strides(), indices);
            std::size_t off_b = indices_to_offset_with_broadcast(out_shape, nb.shape(), nb.strides(), indices);
            out[linear] = (na.data()[off_a] == nb.data()[off_b]);
        }
        return out;
    }

    template <typename A, typename B>
    inline NdArray<bool> neq(const NdArray<A>& a, const NdArray<B>& b)
    {
        using R = typename std::common_type<A, B>::type;
        auto na = ndarray_cast<R>(a);
        auto nb = ndarray_cast<R>(b);
        shape_t out_shape = broadcast_shape(na.shape(), nb.shape());
        NdArray<bool> out(out_shape);
        std::size_t total = out.size();
        shape_t indices;
        for (std::size_t linear = 0; linear < total; ++linear)
        {
            linear_to_indices(linear, out_shape, indices);
            std::size_t off_a = indices_to_offset_with_broadcast(out_shape, na.shape(), na.strides(), indices);
            std::size_t off_b = indices_to_offset_with_broadcast(out_shape, nb.shape(), nb.strides(), indices);
            out[linear] = (na.data()[off_a] != nb.data()[off_b]);
        }
        return out;
    }

    template <typename A, typename B>
    inline NdArray<bool> lt(const NdArray<A>& a, const NdArray<B>& b)
    {
        using R = typename std::common_type<A, B>::type;
        auto na = ndarray_cast<R>(a);
        auto nb = ndarray_cast<R>(b);
        shape_t out_shape = broadcast_shape(na.shape(), nb.shape());
        NdArray<bool> out(out_shape);
        std::size_t total = out.size();
        shape_t indices;
        for (std::size_t linear = 0; linear < total; ++linear)
        {
            linear_to_indices(linear, out_shape, indices);
            std::size_t off_a = indices_to_offset_with_broadcast(out_shape, na.shape(), na.strides(), indices);
            std::size_t off_b = indices_to_offset_with_broadcast(out_shape, nb.shape(), nb.strides(), indices);
            out[linear] = (na.data()[off_a] < nb.data()[off_b]);
        }
        return out;
    }

    template <typename A, typename B>
    inline NdArray<bool> le(const NdArray<A>& a, const NdArray<B>& b)
    {
        using R = typename std::common_type<A, B>::type;
        auto na = ndarray_cast<R>(a);
        auto nb = ndarray_cast<R>(b);
        shape_t out_shape = broadcast_shape(na.shape(), nb.shape());
        NdArray<bool> out(out_shape);
        std::size_t total = out.size();
        shape_t indices;
        for (std::size_t linear = 0; linear < total; ++linear)
        {
            linear_to_indices(linear, out_shape, indices);
            std::size_t off_a = indices_to_offset_with_broadcast(out_shape, na.shape(), na.strides(), indices);
            std::size_t off_b = indices_to_offset_with_broadcast(out_shape, nb.shape(), nb.strides(), indices);
            out[linear] = (na.data()[off_a] <= nb.data()[off_b]);
        }
        return out;
    }

    template <typename A, typename B>
    inline NdArray<bool> gt(const NdArray<A>& a, const NdArray<B>& b)
    {
        using R = typename std::common_type<A, B>::type;
        auto na = ndarray_cast<R>(a);
        auto nb = ndarray_cast<R>(b);
        shape_t out_shape = broadcast_shape(na.shape(), nb.shape());
        NdArray<bool> out(out_shape);
        std::size_t total = out.size();
        shape_t indices;
        for (std::size_t linear = 0; linear < total; ++linear)
        {
            linear_to_indices(linear, out_shape, indices);
            std::size_t off_a = indices_to_offset_with_broadcast(out_shape, na.shape(), na.strides(), indices);
            std::size_t off_b = indices_to_offset_with_broadcast(out_shape, nb.shape(), nb.strides(), indices);
            out[linear] = (na.data()[off_a] > nb.data()[off_b]);
        }
        return out;
    }

    template <typename A, typename B>
    inline NdArray<bool> ge(const NdArray<A>& a, const NdArray<B>& b)
    {
        using R = typename std::common_type<A, B>::type;
        auto na = ndarray_cast<R>(a);
        auto nb = ndarray_cast<R>(b);
        shape_t out_shape = broadcast_shape(na.shape(), nb.shape());
        NdArray<bool> out(out_shape);
        std::size_t total = out.size();
        shape_t indices;
        for (std::size_t linear = 0; linear < total; ++linear)
        {
            linear_to_indices(linear, out_shape, indices);
            std::size_t off_a = indices_to_offset_with_broadcast(out_shape, na.shape(), na.strides(), indices);
            std::size_t off_b = indices_to_offset_with_broadcast(out_shape, nb.shape(), nb.strides(), indices);
            out[linear] = (na.data()[off_a] >= nb.data()[off_b]);
        }
        return out;
    }

} // namespace CNum

#endif // NUMERICALS_UFUNCS_H