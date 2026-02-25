#include "Quadrature.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <limits>

namespace CNum
{
    // Helper: fill 1D Gauss-Legendre points and weights for orders 1..8
    static inline void GetGaussLegendre1D(std::size_t n,
        std::vector<double>& pts,
        std::vector<double>& wts) noexcept
    {
        pts.clear();
        wts.clear();

        switch (n)
        {
        case 1:
            pts = { 0.0 };
            wts = { 2.0 };
            break;
        case 2:
            pts = { -0.5773502691896257, 0.5773502691896257 };
            wts = { 1.0, 1.0 };
            break;
        case 3:
            pts = { -0.7745966692414834, 0.0, 0.7745966692414834 };
            wts = { 0.5555555555555556, 0.8888888888888888, 0.5555555555555556 };
            break;
        case 4:
            pts = { -0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526 };
            wts = { 0.3478548451374539, 0.6521451548625461, 0.6521451548625461, 0.3478548451374539 };
            break;
        case 5:
            pts = { -0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640 };
            wts = { 0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891 };
            break;
        case 6:
            pts = { -0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
                     0.2386191860831969,  0.6612093864662645,  0.9324695142031521 };
            wts = { 0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
                    0.4679139345726910, 0.3607615730481386, 0.1713244923791704 };
            break;
        case 7:
            pts = { -0.9491079123427585, -0.7415311855993944, -0.4058451513773972,
                     0.0,
                     0.4058451513773972,  0.7415311855993944,  0.9491079123427585 };
            wts = { 0.1294849661688697, 0.2797053914892766, 0.3818300505051189,
                    0.4179591836734694, 0.3818300505051189, 0.2797053914892766, 0.1294849661688697 };
            break;
        case 8:
            pts = { -0.9602898564975363, -0.7966664774136267, -0.5255324099163289, -0.1834346424956498,
                     0.1834346424956498,  0.5255324099163289,  0.7966664774136267,  0.9602898564975363 };
            wts = { 0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
                    0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763 };
            break;
        default:
            break;
        }
    }

    // ---- std::vector overload (primary implementation) ----
    void Quadrature::Compute(std::size_t quadorder,
        const std::string& qt,
        std::size_t sdim,
        std::vector<double>& W,
        std::vector<std::vector<double>>& Q)
    {
        std::string type = qt;
        std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

        if (type != "GAUSS" && type != "TRIANGULAR")
            throw std::invalid_argument("Quadrature::Compute: unsupported quadrature type (use \"GAUSS\" or \"TRIANGULAR\")");

        if (sdim < 1 || sdim > 3)
            throw std::invalid_argument("Quadrature::Compute: sdim must be 1, 2 or 3");

        if (type == "GAUSS")
        {
            if (quadorder == 0) quadorder = 1;
            if (quadorder > 8) quadorder = 8;

            std::vector<double> r1pt, r1wt;
            GetGaussLegendre1D(quadorder, r1pt, r1wt);
            if (r1pt.empty() || r1wt.empty()) { W.clear(); Q.clear(); return; }

            const std::size_t n = r1pt.size();
            std::size_t total = 1;
            for (std::size_t i = 0; i < sdim; ++i) total *= n;

            W.assign(total, 0.0);
            Q.resize(total);
            std::size_t idx = 0;
            if (sdim == 1)
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    Q[idx] = { r1pt[i] };
                    W[idx] = r1wt[i];
                    ++idx;
                }
            }
            else if (sdim == 2)
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    for (std::size_t j = 0; j < n; ++j)
                    {
                        Q[idx] = { r1pt[i], r1pt[j] };
                        W[idx] = r1wt[i] * r1wt[j];
                        ++idx;
                    }
                }
            }
            else // sdim == 3
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    for (std::size_t j = 0; j < n; ++j)
                    {
                        for (std::size_t k = 0; k < n; ++k)
                        {
                            Q[idx] = { r1pt[i], r1pt[j], r1pt[k] };
                            W[idx] = r1wt[i] * r1wt[j] * r1wt[k];
                            ++idx;
                        }
                    }
                }
            }
            return;
        }

        // TRIANGULAR
        if (type == "TRIANGULAR")
        {
            if (sdim == 3) // tetrahedra
            {
                if (quadorder != 1 && quadorder != 2 && quadorder != 3) quadorder = 1;

                if (quadorder == 1)
                {
                    Q = { { 0.25, 0.25, 0.25 } };
                    W = { 1.0 / 6.0 };
                    return;
                }
                if (quadorder == 2)
                {
                    Q = { { 0.58541020, 0.13819660, 0.13819660 },
                          { 0.13819660, 0.58541020, 0.13819660 },
                          { 0.13819660, 0.13819660, 0.58541020 },
                          { 0.13819660, 0.13819660, 0.13819660 } };
                    W = { 1.0 / 4.0 / 6.0, 1.0 / 4.0 / 6.0, 1.0 / 4.0 / 6.0, 1.0 / 4.0 / 6.0 };
                    return;
                }
                // quadorder == 3
                Q = { { 0.25, 0.25, 0.25 },
                      { 0.5, 1.0 / 6.0, 1.0 / 6.0 },
                      { 1.0 / 6.0, 0.5, 1.0 / 6.0 },
                      { 1.0 / 6.0, 1.0 / 6.0, 0.5 },
                      { 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0 } };
                W = { -4.0 / 5.0 / 6.0, 9.0 / 20.0 / 6.0, 9.0 / 20.0 / 6.0, 9.0 / 20.0 / 6.0, 9.0 / 20.0 / 6.0 };
                return;
            }
            else // sdim == 2 (triangles)
            {
                if (quadorder == 0) quadorder = 1;
                if (quadorder > 7) quadorder = 1;

                if (quadorder == 1)
                {
                    Q = { { 1.0 / 3.0, 1.0 / 3.0 } };
                    W = { 0.5 };
                    return;
                }
                else if (quadorder == 2)
                {
                    Q = { { 1.0 / 6.0, 1.0 / 6.0 },
                          { 2.0 / 3.0, 1.0 / 6.0 },
                          { 1.0 / 6.0, 2.0 / 3.0 } };
                    W = { 1.0 / 3.0 / 2.0, 1.0 / 3.0 / 2.0, 1.0 / 3.0 / 2.0 };
                    return;
                }
                else if (quadorder <= 5)
                {
                    Q = { { 0.1012865073235, 0.1012865073235 },
                          { 0.7974269853531, 0.1012865073235 },
                          { 0.1012865073235, 0.7974269853531 },
                          { 0.4701420641051, 0.0597158717898 },
                          { 0.4701420641051, 0.4701420641051 },
                          { 0.0597158717898, 0.4701420641051 },
                          { 1.0 / 3.0, 1.0 / 3.0 } };
                    W = { 0.1259391805448 / 2.0, 0.1259391805448 / 2.0, 0.1259391805448 / 2.0,
                          0.1323941527885 / 2.0, 0.1323941527885 / 2.0, 0.1323941527885 / 2.0,
                          0.2250000000000 / 2.0 };
                    return;
                }
                else
                {
                    Q = { { 0.0651301029022, 0.0651301029022 },
                          { 0.8697397941956, 0.0651301029022 },
                          { 0.0651301029022, 0.8697397941956 },
                          { 0.3128654960049, 0.0486903154253 },
                          { 0.6384441885698, 0.3128654960049 },
                          { 0.0486903154253, 0.6384441885698 },
                          { 0.6384441885698, 0.0486903154253 },
                          { 0.3128654960049, 0.6384441885698 },
                          { 0.0486903154253, 0.3128654960049 },
                          { 0.2603459660790, 0.2603459660790 },
                          { 0.4793080678419, 0.2603459660790 },
                          { 0.2603459660790, 0.4793080678419 },
                          { 1.0 / 3.0, 1.0 / 3.0 } };
                    W = { 0.0533472356088 / 2.0, 0.0533472356088 / 2.0, 0.0533472356088 / 2.0,
                          0.0771137608903 / 2.0, 0.0771137608903 / 2.0, 0.0771137608903 / 2.0,
                          0.0771137608903 / 2.0, 0.0771137608903 / 2.0, 0.0771137608903 / 2.0,
                          0.1756152576332 / 2.0, 0.1756152576332 / 2.0, 0.1756152576332 / 2.0,
                          -0.1495700444677 / 2.0 };
                    return;
                }
            }
        }
    }

    // Backwards-compatible GaussQuadrature helpers
    std::vector<QuadPoint2D> GaussQuadrature::GetGaussPoints2D(std::size_t order) noexcept
    {
        try
        {
            std::vector<double> W;
            std::vector<std::vector<double>> Q;
            Quadrature::Compute(order, "GAUSS", 2, W, Q);
            std::vector<QuadPoint2D> out;
            out.reserve(Q.size());
            for (std::size_t i = 0; i < Q.size(); ++i)
            {
                QuadPoint2D qp;
                qp.xi = Q[i][0];
                qp.eta = Q[i][1];
                qp.weight = W[i];
                out.push_back(qp);
            }
            return out;
        }
        catch (...)
        {
            return {};
        }
    }

    double GaussQuadrature::Integrate2D(const std::function<double(double, double)>& f, std::size_t order) noexcept
    {
        if (!f) return std::numeric_limits<double>::quiet_NaN();
        auto qp = GetGaussPoints2D(order);
        if (qp.empty()) return std::numeric_limits<double>::quiet_NaN();
        double sum = 0.0;
        for (const auto& p : qp)
        {
            double val = f(p.xi, p.eta);
            if (!std::isfinite(val)) return std::numeric_limits<double>::quiet_NaN();
            sum += val * p.weight;
        }
        return sum;
    }

    void GaussQuadrature::GetQ4ShapeFunctions(std::size_t order,
        std::vector<std::array<double, 4>>& N,
        std::vector<std::array<double, 4>>& dN_dxi,
        std::vector<std::array<double, 4>>& dN_deta) noexcept
    {
        N.clear();
        dN_dxi.clear();
        dN_deta.clear();

        try
        {
            std::vector<double> W;
            std::vector<std::vector<double>> Q;
            Quadrature::Compute(order, "GAUSS", 2, W, Q);
            if (Q.empty()) return;

            N.reserve(Q.size());
            dN_dxi.reserve(Q.size());
            dN_deta.reserve(Q.size());

            for (std::size_t i = 0; i < Q.size(); ++i)
            {
                const double xi = Q[i][0];
                const double eta = Q[i][1];

                std::array<double, 4> Ni;
                std::array<double, 4> dxi_i;
                std::array<double, 4> deta_i;

                Ni[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
                Ni[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
                Ni[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
                Ni[3] = 0.25 * (1.0 - xi) * (1.0 + eta);

                dxi_i[0] = -0.25 * (1.0 - eta);
                dxi_i[1] = 0.25 * (1.0 - eta);
                dxi_i[2] = 0.25 * (1.0 + eta);
                dxi_i[3] = -0.25 * (1.0 + eta);

                deta_i[0] = -0.25 * (1.0 - xi);
                deta_i[1] = -0.25 * (1.0 + xi);
                deta_i[2] = 0.25 * (1.0 + xi);
                deta_i[3] = 0.25 * (1.0 - xi);

                N.push_back(Ni);
                dN_dxi.push_back(dxi_i);
                dN_deta.push_back(deta_i);
            }
        }
        catch (...)
        {
            N.clear();
            dN_dxi.clear();
            dN_deta.clear();
        }
    }
} // namespace CNum