#ifndef NUMERICALS_SORTING_H
#define NUMERICALS_SORTING_H

#include "Vector.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <cstddef>
#include <vector>
#include <iterator>
#include <type_traits>
#include <limits>
#include <stdexcept>

namespace CNum
{

    // Utility class providing sorting helpers for Vector<T>.
    // All in-place algorithms operate on Vector<T>::begin()/end() so proxy types
    // (e.g. CArray<bool> / std::vector<bool> proxies) remain compatible.
    template <typename T>
    class Sorter
    {
    public:
        using value_type = T;

        // Bubble Sort (in-place)
        template <typename Compare = std::less<T>>
        static void BubbleSort(Vector<T>& v, Compare comp = Compare())
        {
            const std::size_t n = v.size();
            if (n < 2) return;
            for (std::size_t i = 0; i < n - 1; ++i)
            {
                bool swapped = false;
                for (std::size_t j = 0; j < n - 1 - i; ++j)
                {
                    if (comp(v[j + 1], v[j]))
                    {
                        std::swap(v[j], v[j + 1]);
                        swapped = true;
                    }
                }
                if (!swapped) break;
            }
        }

        // Cocktail Sort (bidirectional bubble)
        template <typename Compare = std::less<T>>
        static void CocktailSort(Vector<T>& v, Compare comp = Compare())
        {
            std::size_t n = v.size();
            if (n < 2) return;
            std::size_t start = 0;
            std::size_t end = n - 1;
            bool swapped = true;
            while (swapped)
            {
                swapped = false;
                for (std::size_t i = start; i < end; ++i)
                {
                    if (comp(v[i + 1], v[i]))
                    {
                        std::swap(v[i], v[i + 1]);
                        swapped = true;
                    }
                }
                if (!swapped) break;
                swapped = false;
                if (end == 0) break;
                --end;
                for (std::size_t i = end; i > start; --i)
                {
                    if (comp(v[i], v[i - 1]))
                    {
                        std::swap(v[i], v[i - 1]);
                        swapped = true;
                    }
                }
                ++start;
            }
        }

        // Odd-Even Sort (in-place)
        template <typename Compare = std::less<T>>
        static void OddEvenSort(Vector<T>& v, Compare comp = Compare())
        {
            const std::size_t n = v.size();
            if (n < 2) return;
            bool sorted = false;
            while (!sorted)
            {
                sorted = true;
                // odd phase
                for (std::size_t i = 1; i + 1 <= n - 1; i += 2)
                {
                    if (comp(v[i + 1], v[i]))
                    {
                        std::swap(v[i], v[i + 1]);
                        sorted = false;
                    }
                }
                // even phase
                for (std::size_t i = 0; i + 1 <= n - 1; i += 2)
                {
                    if (comp(v[i + 1], v[i]))
                    {
                        std::swap(v[i], v[i + 1]);
                        sorted = false;
                    }
                }
            }
        }

        // Comb Sort (in-place)
        template <typename Compare = std::less<T>>
        static void CombSort(Vector<T>& v, Compare comp = Compare())
        {
            const std::size_t n = v.size();
            if (n < 2) return;
            double shrink = 1.3;
            std::size_t gap = n;
            bool sorted = false;
            while (!sorted)
            {
                gap = static_cast<std::size_t>(static_cast<double>(gap) / shrink);
                if (gap <= 1)
                {
                    gap = 1;
                    sorted = true;
                }
                std::size_t i = 0;
                while (i + gap < n)
                {
                    if (comp(v[i + gap], v[i]))
                    {
                        std::swap(v[i], v[i + gap]);
                        sorted = false;
                    }
                    ++i;
                }
            }
        }

        // Gnome Sort (in-place)
        template <typename Compare = std::less<T>>
        static void GnomeSort(Vector<T>& v, Compare comp = Compare())
        {
            const std::size_t n = v.size();
            if (n < 2) return;
            std::size_t index = 0;
            while (index < n)
            {
                if (index == 0) ++index;
                if (!comp(v[index], v[index - 1]))
                {
                    ++index;
                }
                else
                {
                    std::swap(v[index], v[index - 1]);
                    --index;
                }
            }
        }

        // QuickSort (in-place) - recursive with median-of-three pivot
        template <typename Compare = std::less<T>>
        static void QuickSort(Vector<T>& v, Compare comp = Compare())
        {
            if (v.size() < 2) return;
            auto begin = v.begin();
            auto end = v.end();

            std::function<void(std::ptrdiff_t, std::ptrdiff_t)> qsort =
                [&](std::ptrdiff_t lo, std::ptrdiff_t hi)
                {
                    if (lo >= hi) return;
                    std::ptrdiff_t mid = lo + (hi - lo) / 2;
                    // median-of-three
                    if (comp(begin[mid], begin[lo])) std::swap(begin[lo], begin[mid]);
                    if (comp(begin[hi], begin[lo])) std::swap(begin[lo], begin[hi]);
                    if (comp(begin[hi], begin[mid])) std::swap(begin[mid], begin[hi]);
                    const T pivot = begin[mid];

                    std::ptrdiff_t i = lo;
                    std::ptrdiff_t j = hi;
                    while (i <= j)
                    {
                        while (comp(begin[i], pivot)) ++i;
                        while (comp(pivot, begin[j])) --j;
                        if (i <= j)
                        {
                            std::swap(begin[i], begin[j]);
                            ++i;
                            --j;
                        }
                    }
                    if (lo < j) qsort(lo, j);
                    if (i < hi) qsort(i, hi);
                };

            qsort(0, static_cast<std::ptrdiff_t>(v.size() - 1));
        }

        // Insertion Sort (in-place)
        template <typename Compare = std::less<T>>
        static void InsertionSort(Vector<T>& v, Compare comp = Compare())
        {
            const std::size_t n = v.size();
            for (std::size_t i = 1; i < n; ++i)
            {
                T key = v[i];
                std::ptrdiff_t j = static_cast<std::ptrdiff_t>(i) - 1;
                while (j >= 0 && comp(key, v[static_cast<std::size_t>(j)]))
                {
                    v[static_cast<std::size_t>(j + 1)] = v[static_cast<std::size_t>(j)];
                    --j;
                }
                v[static_cast<std::size_t>(j + 1)] = key;
            }
        }

        // Shell Sort (in-place) - using Knuth sequence
        template <typename Compare = std::less<T>>
        static void ShellSort(Vector<T>& v, Compare comp = Compare())
        {
            const std::size_t n = v.size();
            std::size_t gap = 1;
            while (gap < n / 3) gap = 3 * gap + 1; // Knuth
            while (gap > 0)
            {
                for (std::size_t i = gap; i < n; ++i)
                {
                    T temp = v[i];
                    std::ptrdiff_t j = static_cast<std::ptrdiff_t>(i);
                    while (j >= static_cast<std::ptrdiff_t>(gap) && comp(temp, v[static_cast<std::size_t>(j - gap)]))
                    {
                        v[static_cast<std::size_t>(j)] = v[static_cast<std::size_t>(j - gap)];
                        j -= static_cast<std::ptrdiff_t>(gap);
                    }
                    v[static_cast<std::size_t>(j)] = temp;
                }
                gap = gap / 3;
            }
        }

        // Selection Sort (in-place)
        template <typename Compare = std::less<T>>
        static void SelectionSort(Vector<T>& v, Compare comp = Compare())
        {
            const std::size_t n = v.size();
            for (std::size_t i = 0; i + 1 < n; ++i)
            {
                std::size_t min_idx = i;
                for (std::size_t j = i + 1; j < n; ++j)
                {
                    if (comp(v[j], v[min_idx])) min_idx = j;
                }
                if (min_idx != i) std::swap(v[i], v[min_idx]);
            }
        }

        // Merge Sort (in-place using temporary buffer)
        template <typename Compare = std::less<T>>
        static void MergeSort(Vector<T>& v, Compare comp = Compare())
        {
            const std::size_t n = v.size();
            if (n < 2) return;
            std::vector<T> temp(n);

            std::function<void(std::size_t, std::size_t)> msort =
                [&](std::size_t lo, std::size_t hi)
                {
                    if (hi - lo <= 1) return;
                    std::size_t mid = lo + (hi - lo) / 2;
                    msort(lo, mid);
                    msort(mid, hi);
                    std::size_t i = lo;
                    std::size_t j = mid;
                    std::size_t k = lo;
                    while (i < mid && j < hi)
                    {
                        if (!comp(v[j], v[i]))
                        {
                            temp[k++] = v[i++];
                        }
                        else
                        {
                            temp[k++] = v[j++];
                        }
                    }
                    while (i < mid) temp[k++] = v[i++];
                    while (j < hi) temp[k++] = v[j++];
                    for (std::size_t t = lo; t < hi; ++t) v[t] = temp[t];
                };

            msort(0, n);
        }

        // Bucket Sort (in-place) - enabled only for arithmetic types
        // Note: This implementation chooses bucket count = max(1, n)
        template <typename Compare = std::less<T>>
        static typename std::enable_if<std::is_arithmetic<T>::value, void>::type
            BucketSort(Vector<T>& v, Compare comp = Compare())
        {
            const std::size_t n = v.size();
            if (n < 2) return;

            T minv = v[0];
            T maxv = v[0];
            for (std::size_t i = 1; i < n; ++i)
            {
                if (comp(v[i], minv)) minv = v[i];
                if (comp(maxv, v[i])) maxv = v[i];
            }
            if (!comp(minv, maxv) && !comp(maxv, minv)) return; // all equal

            std::size_t bucket_count = n;
            std::vector<std::vector<T>> buckets(bucket_count);

            // distribute
            long double range = static_cast<long double>(maxv) - static_cast<long double>(minv);
            if (range == 0.0L)
            {
                // All equal, already handled, but fallback:
                return;
            }
            for (std::size_t i = 0; i < n; ++i)
            {
                long double normalized = (static_cast<long double>(v[i]) - static_cast<long double>(minv)) / range;
                std::size_t idx = static_cast<std::size_t>(normalized * (bucket_count - 1));
                buckets[idx].push_back(v[i]);
            }

            // sort buckets and concatenate
            std::size_t pos = 0;
            for (std::size_t b = 0; b < bucket_count; ++b)
            {
                if (!buckets[b].empty())
                {
                    std::sort(buckets[b].begin(), buckets[b].end(), comp);
                    for (std::size_t k = 0; k < buckets[b].size(); ++k)
                    {
                        v[pos++] = buckets[b][k];
                    }
                }
            }
        }

        // Heap Sort (in-place) - uses std::make_heap / std::sort_heap
        template <typename Compare = std::less<T>>
        static void HeapSort(Vector<T>& v, Compare comp = Compare())
        {
            if (v.size() < 2) return;
            std::make_heap(v.begin(), v.end(), comp);
            std::sort_heap(v.begin(), v.end(), comp);
        }
    };

} // namespace CNum

#endif // NUMERICALS_SORTING_H