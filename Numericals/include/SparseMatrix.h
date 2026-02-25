#pragma once

#ifndef NUMERICALS_SPARSEMATRIX_H
#define NUMERICALS_SPARSEMATRIX_H

#include "Vector.h"
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <cstddef>
#include <cstring>
#include <numeric>

namespace CNum
{

    // CSR sparse matrix with two assembly modes:
    //   1. Triplet-based: AddValue() + Finalize()  (original, convenient, slower)
    //   2. Structure-prealloc: PreallocateStructure() + AddValueFast() + FinalizeFast()
    //      (O(nnz) assembly with no sort — Abaqus-style)
    template <typename T>
    class SparseMatrix
    {
    public:
        using value_type = T;

        SparseMatrix(std::size_t rows = 0, std::size_t cols = 0) noexcept
            : rows_(rows), cols_(cols), finalized_(false)
        {
        }

        void Resize(std::size_t rows, std::size_t cols) noexcept
        {
            rows_ = rows;
            cols_ = cols;
            entries_.clear();
            finalized_ = false;
            values_.clear();
            colIndex_.clear();
            rowPtr_.clear();
            rowMap_.clear();
        }

        std::size_t Rows() const noexcept { return rows_; }
        std::size_t Cols() const noexcept { return cols_; }

        // =====================================================================
        //  MODE 1: Triplet assembly (original API, unchanged)
        // =====================================================================

        void AddValue(std::size_t r, std::size_t c, T val)
        {
            if (r >= rows_ || c >= cols_) throw std::out_of_range("AddValue: index out of range");
            entries_.emplace_back(r, c, val);
            finalized_ = false;
        }

        void Reserve(std::size_t nnz) { entries_.reserve(nnz); }

        void Finalize()
        {
            std::sort(entries_.begin(), entries_.end(),
                [](const Triplet& a, const Triplet& b) {
                    if (a.r != b.r) return a.r < b.r;
                    return a.c < b.c;
                });

            values_.clear();
            colIndex_.clear();
            rowPtr_.assign(rows_ + 1, 0);

            for (std::size_t i = 0; i < entries_.size(); )
            {
                std::size_t r = entries_[i].r;
                std::size_t c = entries_[i].c;
                T sum = entries_[i].v;
                ++i;
                while (i < entries_.size() && entries_[i].r == r && entries_[i].c == c)
                {
                    sum += entries_[i].v;
                    ++i;
                }
                values_.push_back(sum);
                colIndex_.push_back(c);
                ++rowPtr_[r + 1];
            }

            for (std::size_t r = 0; r < rows_; ++r)
                rowPtr_[r + 1] += rowPtr_[r];

            finalized_ = true;
        }

        // =====================================================================
        //  MODE 2: Structure-prealloc assembly (O(nnz), no sort)
        //  Usage:
        //    1. Call PreallocateStructure(elementConnectivity, ...) once
        //    2. For each element, call AddValueFast(row, col, val) — O(1) amortized
        //    3. Call FinalizeFast() — just sets finalized_ = true (structure already built)
        // =====================================================================

        // Build CSR structure from element connectivity.
        // elemDofs[e] = vector of global DOF indices for element e.
        // This determines the sparsity pattern and creates a (row, col) -> CSR-offset lookup.
        void PreallocateStructure(const std::vector<std::vector<std::size_t>>& elemDofs)
        {
            // Step 1: Count non-zeros per row using sorted sets
            // For each row, collect all unique column indices
            std::vector<std::vector<std::size_t>> rowCols(rows_);
            for (const auto& dofs : elemDofs)
            {
                for (std::size_t di : dofs)
                {
                    if (di >= rows_) continue;
                    for (std::size_t dj : dofs)
                    {
                        if (dj >= cols_) continue;
                        rowCols[di].push_back(dj);
                    }
                }
            }

            // Sort and unique each row
            for (std::size_t r = 0; r < rows_; ++r)
            {
                auto& rc = rowCols[r];
                std::sort(rc.begin(), rc.end());
                rc.erase(std::unique(rc.begin(), rc.end()), rc.end());
            }

            // Build CSR arrays
            rowPtr_.resize(rows_ + 1);
            rowPtr_[0] = 0;
            std::size_t nnz = 0;
            for (std::size_t r = 0; r < rows_; ++r)
            {
                nnz += rowCols[r].size();
                rowPtr_[r + 1] = nnz;
            }

            colIndex_.resize(nnz);
            values_.assign(nnz, T(0));

            for (std::size_t r = 0; r < rows_; ++r)
            {
                std::size_t offset = rowPtr_[r];
                for (std::size_t k = 0; k < rowCols[r].size(); ++k)
                {
                    colIndex_[offset + k] = rowCols[r][k];
                }
            }

            // Build per-row binary-searchable lookup (columns are already sorted)
            // rowMap_ is not needed — we use binary search on colIndex_ within [rowPtr_[r], rowPtr_[r+1])
            rowMap_.clear(); // unused for this path
            finalized_ = false; // will be set by FinalizeFast
        }

        // O(log(nnz_per_row)) insertion into pre-allocated CSR.
        // Columns within each row must already be sorted (guaranteed by PreallocateStructure).
        void AddValueFast(std::size_t r, std::size_t c, T val)
        {
            std::size_t start = rowPtr_[r];
            std::size_t end = rowPtr_[r + 1];
            // Binary search for column c in colIndex_[start..end)
            const std::size_t* base = colIndex_.data() + start;
            std::size_t len = end - start;
            // Standard lower_bound on raw pointer
            const std::size_t* it = std::lower_bound(base, base + len, c);
            std::size_t idx = static_cast<std::size_t>(it - colIndex_.data());
            // Bounds check (should always find if structure was built correctly)
            if (idx < end && colIndex_[idx] == c)
            {
                values_[idx] += val;
            }
            // else: structural zero, silently ignored (or could throw)
        }

        void FinalizeFast() noexcept
        {
            finalized_ = true;
            entries_.clear(); // free triplet memory if any
            entries_.shrink_to_fit();
        }

        // =====================================================================
        //  Query / SpMV
        // =====================================================================

        std::size_t NonZeros() const noexcept { return values_.size(); }

        // High-performance SpMV using raw double* pointers.
        // This is the critical inner loop of CG — must be as fast as possible.
        void MatVecRaw(const double* __restrict x, double* __restrict y) const noexcept
        {
            const double* __restrict vp = values_.data();
            const std::size_t* __restrict cp = colIndex_.data();
            const std::size_t* __restrict rp = rowPtr_.data();

            for (std::size_t r = 0; r < rows_; ++r)
            {
                const std::size_t rs = rp[r];
                const std::size_t re = rp[r + 1];
                double sum = 0.0;
                for (std::size_t k = rs; k < re; ++k)
                {
                    sum += vp[k] * x[cp[k]];
                }
                y[r] = sum;
            }
        }

        // Original MatVec (kept for backward compatibility)
        template <typename U>
        void MatVec(const Vector<U>& x, Vector<typename std::common_type<T, U, double>::type>& out) const
        {
            using R = typename std::common_type<T, U, double>::type;
            if (!finalized_) throw std::runtime_error("SparseMatrix: call Finalize() before MatVec");
            if (x.size() != cols_) throw std::runtime_error("MatVec: size mismatch");
            if (out.size() != rows_) out = Vector<R>(rows_, R());

            // Use raw pointer path for double types
            MatVecRaw(x.data(), out.data());
        }

        const std::vector<T>& Values() const noexcept { return values_; }
        const std::vector<std::size_t>& ColIndex() const noexcept { return colIndex_; }
        const std::vector<std::size_t>& RowPtr() const noexcept { return rowPtr_; }
        bool IsFinalized() const noexcept { return finalized_; }

        // Extract diagonal values into caller-provided array
        void ExtractDiagonal(std::vector<T>& diag) const
        {
            diag.assign(rows_, T(0));
            for (std::size_t r = 0; r < rows_; ++r)
            {
                for (std::size_t k = rowPtr_[r]; k < rowPtr_[r + 1]; ++k)
                {
                    if (colIndex_[k] == r) { diag[r] = values_[k]; break; }
                }
            }
        }

        // Compute IC(0) factorization on the existing CSR structure (in-place on a copy of values).
        // Returns the IC(0) values array. The sparsity pattern (colIndex_, rowPtr_) is shared.
        // Assumes SPD matrix, same structure for L (lower triangle stored in CSR).
        // For IC(0): L has the same sparsity as lower(A). We store the full factor.
        std::vector<T> ComputeIC0() const
        {
            std::vector<T> Lval(values_);

            // Helper: find CSR offset for entry (row, col), or return size_t(-1) if not found
            auto findOffset = [&](std::size_t row, std::size_t col) -> std::size_t {
                const std::size_t* base = colIndex_.data() + rowPtr_[row];
                std::size_t len = rowPtr_[row + 1] - rowPtr_[row];
                const std::size_t* it = std::lower_bound(base, base + len, col);
                std::size_t idx = static_cast<std::size_t>(it - colIndex_.data());
                if (idx < rowPtr_[row + 1] && colIndex_[idx] == col)
                    return idx;
                return static_cast<std::size_t>(-1);
                };

            for (std::size_t i = 0; i < rows_; ++i)
            {
                std::size_t ii_start = rowPtr_[i];
                std::size_t ii_end = rowPtr_[i + 1];

                // Collect lower-triangular column indices for row i
                // and compute L(i,j) for each j < i
                for (std::size_t kij = ii_start; kij < ii_end; ++kij)
                {
                    std::size_t j = colIndex_[kij];
                    if (j >= i) break; // past lower triangle

                    // L(i,j) = A(i,j) - sum_{k < j} L(i,k) * L(j,k)
                    T sum = T(0);
                    for (std::size_t kik = ii_start; kik < ii_end; ++kik)
                    {
                        std::size_t k = colIndex_[kik];
                        if (k >= j) break;
                        // Find L(j,k) in row j
                        std::size_t jk_off = findOffset(j, k);
                        if (jk_off != static_cast<std::size_t>(-1))
                            sum += Lval[kik] * Lval[jk_off];
                    }

                    // Find diagonal L(j,j)
                    std::size_t jj_off = findOffset(j, j);
                    T ljj = (jj_off != static_cast<std::size_t>(-1)) ? Lval[jj_off] : T(1);
                    if (std::abs(ljj) < std::numeric_limits<T>::epsilon() * T(1e4))
                        ljj = T(1);

                    Lval[kij] = (Lval[kij] - sum) / ljj;
                }

                // Diagonal: L(i,i) = sqrt( A(i,i) - sum_{k < i} L(i,k)^2 )
                std::size_t diag_off = findOffset(i, i);
                if (diag_off != static_cast<std::size_t>(-1))
                {
                    T sum = T(0);
                    for (std::size_t kik = ii_start; kik < ii_end; ++kik)
                    {
                        std::size_t k = colIndex_[kik];
                        if (k >= i) break;
                        sum += Lval[kik] * Lval[kik];
                    }
                    T diagVal = Lval[diag_off] - sum;
                    Lval[diag_off] = (diagVal > T(0)) ? std::sqrt(diagVal) : T(1);
                }
            }

            return Lval;
        }

        // Forward/backward solve with IC(0) factor: solve L L^T z = r
        // L stored in CSR with same structure as A. Lval from ComputeIC0().
        void IC0Solve(const std::vector<T>& Lval, const double* __restrict r, double* __restrict z) const noexcept
        {
            // Forward solve: L y = r
            // y[i] = ( r[i] - sum_{j<i} L(i,j)*y[j] ) / L(i,i)
            std::vector<double> y(rows_);
            for (std::size_t i = 0; i < rows_; ++i)
            {
                double sum = static_cast<double>(r[i]);
                double diag = 1.0;
                for (std::size_t k = rowPtr_[i]; k < rowPtr_[i + 1]; ++k)
                {
                    std::size_t c = colIndex_[k];
                    if (c < i)
                        sum -= static_cast<double>(Lval[k]) * y[c];
                    else if (c == i)
                    {
                        diag = static_cast<double>(Lval[k]);
                        break;
                    }
                }
                y[i] = (std::abs(diag) > 1e-30) ? (sum / diag) : sum;
            }

            // Backward solve: L^T z = y
            // z[i] = ( y[i] - sum_{j>i where L(j,i) exists} L(j,i)*z[j] ) / L(i,i)
            //
            // Process rows in reverse. Use scatter approach:
            // After computing z[i], for each j < i where L(i,j) is in the sparsity,
            // subtract L(i,j)*z[i] from y[j].
            for (std::size_t i = rows_; i-- > 0;)
            {
                // Find diagonal
                double diag = 1.0;
                for (std::size_t k = rowPtr_[i]; k < rowPtr_[i + 1]; ++k)
                {
                    if (colIndex_[k] == i)
                    {
                        diag = static_cast<double>(Lval[k]);
                        break;
                    }
                }

                double val = (std::abs(diag) > 1e-30) ? (y[i] / diag) : y[i];
                z[i] = val;

                // Scatter: for each L(i,j) with j < i, update y[j] -= L(i,j) * z[i]
                for (std::size_t k = rowPtr_[i]; k < rowPtr_[i + 1]; ++k)
                {
                    std::size_t c = colIndex_[k];
                    if (c < i)
                        y[c] -= static_cast<double>(Lval[k]) * val;
                    else
                        break;
                }
            }
        }

    private:
        struct Triplet { std::size_t r, c; T v; Triplet(std::size_t rr, std::size_t cc, T vv) : r(rr), c(cc), v(vv) {} };
        std::size_t rows_{ 0 }, cols_{ 0 };
        bool finalized_{ false };

        std::vector<Triplet> entries_;
        std::vector<T> values_;
        std::vector<std::size_t> colIndex_;
        std::vector<std::size_t> rowPtr_;
        std::vector<std::size_t> rowMap_; // reserved for future use
    };

} // namespace CNum

#endif // NUMERICALS_SPARSEMATRIX_H