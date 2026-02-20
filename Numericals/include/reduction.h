#ifndef NUMERICALS_REDUCTIONS_H
#define NUMERICALS_REDUCTIONS_H

#include "dtypes.h"
#include "CArray.h"
#include "views.h"

#include <numeric>
#include <type_traits>

namespace numericals
{

    // Sum reduction over any type that supports size() and operator[]
    template <typename Container>
    typename std::decay<typename Container::value_type>::type sum(const Container& c)
    {
        using T = typename std::decay<typename Container::value_type>::type;
        T acc = T();
        std::size_t n = c.size();
        for (std::size_t i = 0; i < n; ++i)
            acc += c[i];
        return acc;
    }

    // Mean reduction
    template <typename Container>
    double mean(const Container& c)
    {
        using T = typename std::decay<typename Container::value_type>::type;
        std::size_t n = c.size();
        if (n == 0) throw std::runtime_error("mean: empty container");
        T s = sum(c);
        return static_cast<double>(s) / static_cast<double>(n);
    }

} // namespace numericals

#endif // NUMERICALS_REDUCTIONS_H