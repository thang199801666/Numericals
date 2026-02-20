# CONTRIBUTING

## Guidelines

This project provides a small numerical array library (CArray and Vector) and numerical utilities (NonlinearEquations). When adding new API surface, follow these conventions:

- Public APIs must be well-documented in code comments and maintain consistency with existing semantics.
- Public API names use PascalCase for function and method names (e.g., `Find`, `Contains`, `LowerBoundIndex`) to match the project's naming convention.
- Prefer `noexcept` for functions that do not throw (e.g., `Find`, `Contains`, `FindIf`, `LowerBoundIndex`, `UpperBoundIndex`, `BinarySearch`).
- Use `std::algorithm` utilities where appropriate (e.g., `std::find`, `std::lower_bound`, `std::binary_search`).
- For value-not-found semantics prefer returning `std::size_t` with a sentinel `npos` (defined as `static constexpr std::size_t npos = static_cast<std::size_t>(-1);`) for non-throwing lookups. Provide an `IndexOf` variant that throws `std::runtime_error` when the value is not found, for callers that prefer exceptions.
- For predicate-based searches provide `FindIf` accepting any callable that returns a convertible-to-bool type.
- When adding search helpers to `Vector<T>`, ensure they operate on the underlying data via `begin()`/`end()` pointers to preserve compatibility with proxy types.

## Coding style

- Follow existing file formatting (4 spaces indentation inside namespaces/classes). Maintain the existing naming conventions (PascalCase for public member functions such as `FindIf`, `LowerBoundIndex`).
- Include `#include <algorithm>` in headers where algorithms are used.

## Tests

- Add unit tests for new search helpers covering empty vectors, single-element vectors, missing values, multiple matches, and sorted/unsorted cases for `BinarySearch` and `LowerBoundIndex`/`UpperBoundIndex`.

## API additions (example)

Add the following utilities to `Vector<T>`:

- `static constexpr std::size_t npos`
- `std::size_t Find(const T& value) const noexcept`
- `std::size_t IndexOf(const T& value) const` (throws `std::runtime_error` if not found)
- `bool Contains(const T& value) const noexcept`
- `template <typename Pred> std::size_t FindIf(Pred&& pred) const noexcept`