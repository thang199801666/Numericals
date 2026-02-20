#pragma once
#ifndef NUMERICALS_SEARCHING_H
#define NUMERICALS_SEARCHING_H

#include "vector.h"
#include <algorithm>
#include <functional>
#include <type_traits>
#include <stdexcept>
#include <limits>
#include <iterator>

namespace CNum
{

    // Utility class providing search helpers for Vector<T>.
    // Conventions:
    // - Non-throwing lookups return std::size_t with sentinel `npos`.
    // - Index-returning variants that prefer exceptions are provided as throwing `IndexOf` / value-returning functions.
    template <typename T>
    class Search
    {
    public:
        using value_type = T;
        static constexpr std::size_t npos = static_cast<std::size_t>(-1);

        // Linear search (returns index or `npos`), noexcept.
        static std::size_t LinearSearch(const Vector<T>& v, const T& value) noexcept
        {
            auto it = std::find(v.begin(), v.end(), value);
            if (it == v.end()) return npos;
            return static_cast<std::size_t>(std::distance(v.begin(), it));
        }

        // IndexOf: throws std::runtime_error if not found (per project conventions).
        static std::size_t IndexOf(const Vector<T>& v, const T& value)
        {
            auto idx = LinearSearch(v, value);
            if (idx == npos) throw std::runtime_error("IndexOf: value not found");
            return idx;
        }

        // Binary search (returns true if found). Expects sorted ascending with provided comparator.
        template <typename Compare = std::less<T>>
        static bool BinarySearch(const Vector<T>& v, const T& value, Compare comp = Compare()) noexcept
        {
            return !v.empty() && std::binary_search(v.begin(), v.end(), value, comp);
        }

        // BinarySearchIndex: returns index if found, npos otherwise. Expects sorted ascending.
        template <typename Compare = std::less<T>>
        static std::size_t BinarySearchIndex(const Vector<T>& v, const T& value, Compare comp = Compare()) noexcept
        {
            if (v.empty()) return npos;
            auto it = std::lower_bound(v.begin(), v.end(), value, comp);
            if (it == v.end()) return npos;
            // equality via comparator: neither a<b nor b<a
            if (comp(*it, value) || comp(value, *it)) return npos;
            return static_cast<std::size_t>(std::distance(v.begin(), it));
        }

        // Interpolation search: enabled only for arithmetic types.
        // Returns index or npos. Assumes ascending-sorted data.
        template <typename U = T>
        static typename std::enable_if<std::is_arithmetic<U>::value, std::size_t>::type
            InterpolationSearch(const Vector<T>& v, const T& value) noexcept
        {
            if (v.empty()) return npos;
            std::size_t lo = 0;
            std::size_t hi = v.size() - 1;

            while (lo <= hi &&
                static_cast<long double>(value) >= static_cast<long double>(v[lo]) &&
                static_cast<long double>(value) <= static_cast<long double>(v[hi]))
            {
                long double loVal = static_cast<long double>(v[lo]);
                long double hiVal = static_cast<long double>(v[hi]);
                if (hiVal == loVal)
                {
                    if (static_cast<long double>(v[lo]) == static_cast<long double>(value)) return lo;
                    return npos;
                }

                long double fraction = (static_cast<long double>(value) - loVal) / (hiVal - loVal);
                std::size_t pos = lo + static_cast<std::size_t>((hi - lo) * fraction);
                if (pos >= v.size()) return npos;

                if (v[pos] == value) return pos;
                if (static_cast<long double>(v[pos]) < static_cast<long double>(value))
                {
                    lo = pos + 1;
                }
                else
                {
                    if (pos == 0) break;
                    hi = pos - 1;
                }
            }

            if (lo < v.size() && v[lo] == value) return lo;
            return npos;
        }

        // Index of maximum element (returns npos if empty), noexcept.
        static std::size_t MaxIndex(const Vector<T>& v) noexcept
        {
            if (v.empty()) return npos;
            auto it = std::max_element(v.begin(), v.end());
            return static_cast<std::size_t>(std::distance(v.begin(), it));
        }

        // Index of minimum element (returns npos if empty), noexcept.
        static std::size_t MinIndex(const Vector<T>& v) noexcept
        {
            if (v.empty()) return npos;
            auto it = std::min_element(v.begin(), v.end());
            return static_cast<std::size_t>(std::distance(v.begin(), it));
        }

        // MaxValue / MinValue: throw when container empty (index-returning variants are non-throwing above).
        static T MaxValue(const Vector<T>& v)
        {
            auto idx = MaxIndex(v);
            if (idx == npos) throw std::runtime_error("MaxValue: empty container");
            return v[idx];
        }

        static T MinValue(const Vector<T>& v)
        {
            auto idx = MinIndex(v);
            if (idx == npos) throw std::runtime_error("MinValue: empty container");
            return v[idx];
        }

        // N-th largest (1-based). Throws std::runtime_error for invalid n.
        static T NthLargestValue(Vector<T> v, std::size_t n)
        {
            if (n == 0 || n > v.size()) throw std::runtime_error("NthLargestValue: n out of range");
            std::nth_element(v.begin(), v.begin() + (n - 1), v.end(), std::greater<T>());
            return v[n - 1];
        }

        // M-th smallest (1-based). Throws std::runtime_error for invalid m.
        static T MthSmallestValue(Vector<T> v, std::size_t m)
        {
            if (m == 0 || m > v.size()) throw std::runtime_error("MthSmallestValue: m out of range");
            std::nth_element(v.begin(), v.begin() + (m - 1), v.end(), std::less<T>());
            return v[m - 1];
        }

        // Check if sorted ascending (noexcept).
        template <typename Compare = std::less<T>>
        static bool IsSorted(const Vector<T>& v, Compare comp = Compare()) noexcept
        {
            return std::is_sorted(v.begin(), v.end(), comp);
        }

        // Check if sorted descending (noexcept).
        template <typename Compare = std::greater<T>>
        static bool IsSortedDescending(const Vector<T>& v, Compare comp = Compare()) noexcept
        {
            return std::is_sorted(v.begin(), v.end(), comp);
        }
    };

} // namespace CNum

#endif // NUMERICALS_SEARCHING_H