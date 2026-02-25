#include "TestFEM2D.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <regex>
#include "Vector.h"
#include "Matrix.h"
#include "Quadrature.h"
#include "LinearEquations.h"
#include "SparseMatrix.h"
#include "IterativeSolvers.h"

static inline std::string Trim(const std::string& s)
{
    std::size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    std::size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

bool ParseAbaqusInp(const std::string& filename,
    std::vector<std::array<double, 2>>& nodes,
    std::vector<std::array<std::size_t, 4>>& elems,
    std::unordered_map<std::string, std::vector<std::size_t>>& nsets,
    double& outPressure,
    double& outE,
    double& outNu) noexcept
{
    nodes.clear();
    elems.clear();
    nsets.clear();
    outPressure = 0.0;
    outE = 2.1e11;
    outNu = 0.3;

    try
    {
        std::ifstream ifs(filename);
        if (!ifs.is_open()) return false;

        // Read entire file into a string for regex processing
        std::string content((std::istreambuf_iterator<char>(ifs)),
            std::istreambuf_iterator<char>());
        ifs.close();

        // ---- Regex patterns ----

        // Node line: id, x, y [, z]
        //   e.g. "  42,  0.500000,  1.000000, 0.0"
        static const std::regex reNode(
            R"(\n\s*(\d+)\s*,\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*,\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?))");

        // Element line: id, n1, n2, n3, n4
        //   e.g. "  1, 1, 2, 13, 12"
        static const std::regex reElem(
            R"(\n\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+))");

        // *Nset keyword with nset=Name and optional generate
        //   e.g. "*Nset, nset=LeftEdge, generate"
        static const std::regex reNsetKeyword(
            R"(\n\*[Nn][Ss][Ee][Tt]\b[^,]*,\s*[Nn][Ss][Ee][Tt]\s*=\s*(\S+?)(?:\s*,\s*([Gg][Ee][Nn][Ee][Rr][Aa][Tt][Ee]))?\s*(?=\n))");

        // *Elastic data line: E, nu
        //   e.g. "210000000000, 0.3"
        static const std::regex reElastic(
            R"(\*[Ee][Ll][Aa][Ss][Tt][Ii][Cc]\b[^\n]*\n\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*,\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?))");

        // *Dload data line: [elem,] P, magnitude
        //   e.g. "P, 1000"  or  ", P, 1000"
        static const std::regex reDload(
            R"(\*[Dd][Ll][Oo][Aa][Dd]\b[^\n]*\n[^\n]*?([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*(?=\n|$))");

        // Generate data line: start, end, step
        static const std::regex reGenerate(
            R"(\n\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d+))?\s*(?=\n|$))");

        // Comma-separated integer list (for explicit nset data)
        static const std::regex reIntList(R"(\d+)");

        // ---- Extract *Elastic material properties ----
        {
            std::smatch m;
            if (std::regex_search(content, m, reElastic))
            {
                outE = std::stod(m[1].str());
                outNu = std::stod(m[2].str());
            }
        }

        // ---- Extract *Dload pressure ----
        {
            std::smatch m;
            if (std::regex_search(content, m, reDload))
            {
                outPressure = std::stod(m[1].str());
            }
        }

        // ---- Extract node coordinates from the *Node block ----
        // Find the *Node section boundaries
        std::unordered_map<std::size_t, std::pair<double, double>> tmpNodes;
        {
            static const std::regex reNodeSection(
                R"(\*Node\b[^\n]*\n([\s\S]*?)(?=\*[A-Za-z]))",
                std::regex::icase);
            std::smatch secMatch;
            if (std::regex_search(content, secMatch, reNodeSection))
            {
                std::string nodeBlock = secMatch[1].str();
                auto it = std::sregex_iterator(nodeBlock.begin(), nodeBlock.end(), reNode);
                auto end = std::sregex_iterator();
                for (; it != end; ++it)
                {
                    std::size_t id = static_cast<std::size_t>(std::stoull((*it)[1].str()));
                    double x = std::stod((*it)[2].str());
                    double y = std::stod((*it)[3].str());
                    tmpNodes[id] = std::make_pair(x, y);
                }
            }
        }
        if (tmpNodes.empty()) return false;

        // ---- Extract element connectivity from the *Element block ----
        {
            static const std::regex reElemSection(
                R"(\*Element\b[^\n]*\n([\s\S]*?)(?=\*[A-Za-z]))",
                std::regex::icase);
            std::smatch secMatch;
            if (std::regex_search(content, secMatch, reElemSection))
            {
                std::string elemBlock = secMatch[1].str();
                auto it = std::sregex_iterator(elemBlock.begin(), elemBlock.end(), reElem);
                auto end = std::sregex_iterator();
                for (; it != end; ++it)
                {
                    std::array<std::size_t, 4> el = {
                        static_cast<std::size_t>(std::stoull((*it)[2].str())),
                        static_cast<std::size_t>(std::stoull((*it)[3].str())),
                        static_cast<std::size_t>(std::stoull((*it)[4].str())),
                        static_cast<std::size_t>(std::stoull((*it)[5].str()))
                    };
                    elems.push_back(el);
                }
            }
        }

        // ---- Extract *Nset definitions ----
        {
            auto it = std::sregex_iterator(content.begin(), content.end(), reNsetKeyword);
            auto end = std::sregex_iterator();
            for (; it != end; ++it)
            {
                std::string nsetName = (*it)[1].str();
                // Remove trailing comma if present from name capture
                if (!nsetName.empty() && nsetName.back() == ',')
                    nsetName.pop_back();
                nsetName = Trim(nsetName);

                bool isGenerate = (*it)[2].matched;
                std::size_t keywordEnd = static_cast<std::size_t>(it->position() + it->length());

                // Find the data lines: everything from keyword end to the next * keyword
                std::size_t dataEnd = content.find("\n*", keywordEnd);
                if (dataEnd == std::string::npos) dataEnd = content.size();
                std::string dataBlock = content.substr(keywordEnd, dataEnd - keywordEnd);

                std::vector<std::size_t>& vec = nsets[nsetName];

                if (isGenerate)
                {
                    // Parse "start, end[, step]" lines
                    auto git = std::sregex_iterator(dataBlock.begin(), dataBlock.end(), reGenerate);
                    auto gend = std::sregex_iterator();
                    for (; git != gend; ++git)
                    {
                        std::size_t start = static_cast<std::size_t>(std::stoull((*git)[1].str()));
                        std::size_t stop = static_cast<std::size_t>(std::stoull((*git)[2].str()));
                        std::size_t step = (*git)[3].matched
                            ? static_cast<std::size_t>(std::stoull((*git)[3].str()))
                            : 1;
                        if (step == 0) step = 1;
                        for (std::size_t id = start; id <= stop; id += step)
                            vec.push_back(id);
                    }
                }
                else
                {
                    // Parse explicit comma-separated integer lists
                    auto nit = std::sregex_iterator(dataBlock.begin(), dataBlock.end(), reIntList);
                    auto nend = std::sregex_iterator();
                    for (; nit != nend; ++nit)
                        vec.push_back(static_cast<std::size_t>(std::stoull(nit->str())));
                }
            }
        }

        // ---- Build 0-based node array and ID mapping ----
        std::vector<std::size_t> ids;
        ids.reserve(tmpNodes.size());
        for (auto& kv : tmpNodes) ids.push_back(kv.first);
        std::sort(ids.begin(), ids.end());

        std::unordered_map<std::size_t, std::size_t> idToIndex;
        nodes.resize(ids.size());
        for (std::size_t i = 0; i < ids.size(); ++i)
        {
            idToIndex[ids[i]] = i;
            nodes[i][0] = tmpNodes[ids[i]].first;
            nodes[i][1] = tmpNodes[ids[i]].second;
        }

        // Remap element connectivity to 0-based
        std::vector<std::array<std::size_t, 4>> elems0;
        elems0.reserve(elems.size());
        for (auto& e : elems)
        {
            std::array<std::size_t, 4> elIdx;
            for (int k = 0; k < 4; ++k)
            {
                auto it = idToIndex.find(e[k]);
                if (it == idToIndex.end()) return false;
                elIdx[k] = it->second;
            }
            elems0.push_back(elIdx);
        }
        elems.swap(elems0);

        // Remap nset IDs to 0-based
        for (auto& kv : nsets)
        {
            for (auto& nid : kv.second)
            {
                auto it = idToIndex.find(nid);
                if (it != idToIndex.end())
                    nid = it->second;
            }
        }

        return true;
    }
    catch (...)
    {
        return false;
    }
}

namespace CNum
{
    std::vector<double> SolveMindlinReissner_Unstructured(
        const std::vector<std::array<double, 2>>& nodes,
        const std::vector<std::array<std::size_t, 4>>& elems,
        const std::vector<std::size_t>& boundaryNodeIds_0based,
        const std::function<double(double, double)>& qfunc,
        double E, double nu, double t,
        double boundaryValueW = 0.0,
        double tol = 1e-10,
        std::size_t quadOrder = 2)
    {
        if (nodes.empty() || elems.empty()) throw std::invalid_argument("nodes and elems must be non-empty");
        if (!(t > 0.0)) throw std::invalid_argument("thickness must be > 0.");

        const std::size_t numNodes = nodes.size();
        const std::size_t numElems = elems.size();

        const std::size_t dofPerNode = 5;
        const std::size_t totalDofs = numNodes * dofPerNode;

        const double G = E / (2.0 * (1.0 + nu));
        const double kappa = 5.0 / 6.0;

        const double factorM = E * t / (1.0 - nu * nu);
        double D_m[3][3] = {
            { factorM * 1.0,        factorM * nu,      0.0 },
            { factorM * nu,         factorM * 1.0,     0.0 },
            { 0.0,                  0.0,               factorM * (1.0 - nu) / 2.0 }
        };

        const double factorB = E * t * t * t / (12.0 * (1.0 - nu * nu));
        double D_b[3][3] = {
            { factorB * 1.0,        factorB * nu,      0.0 },
            { factorB * nu,         factorB * 1.0,     0.0 },
            { 0.0,                  0.0,               factorB * (1.0 - nu) / 2.0 }
        };

        const double factorS = kappa * G * t;

        std::vector<double> W;
        std::vector<std::vector<double>> Q;
        Quadrature::Compute(quadOrder, "GAUSS", 2, W, Q);
        if (Q.empty() || W.empty() || Q.size() != W.size())
            throw std::invalid_argument("Unsupported quadrature order or Compute() returned empty data");
        const std::size_t nqp = Q.size();

        std::vector<std::array<double, 4>> Nref;
        std::vector<std::array<double, 4>> dN_dxi_ref;
        std::vector<std::array<double, 4>> dN_deta_ref;
        GaussQuadrature::GetQ4ShapeFunctions(quadOrder, Nref, dN_dxi_ref, dN_deta_ref);
        if (Nref.empty() || dN_dxi_ref.empty() || dN_deta_ref.empty())
            throw std::invalid_argument("GetQ4ShapeFunctions returned empty data");
        if (Nref.size() != Q.size())
            throw std::invalid_argument("Shape function count does not match quadrature point count");

        const std::size_t nposVal = static_cast<std::size_t>(-1);
        std::vector<char> isBoundary(totalDofs, 0);
        std::vector<double> uPrescribed(totalDofs, 0.0);

        for (std::size_t id : boundaryNodeIds_0based)
        {
            if (id >= numNodes) continue;
            std::size_t base = id * dofPerNode;
            isBoundary[base + 0] = 1; uPrescribed[base + 0] = 0.0;
            isBoundary[base + 1] = 1; uPrescribed[base + 1] = 0.0;
            isBoundary[base + 2] = 1; uPrescribed[base + 2] = boundaryValueW;
            isBoundary[base + 3] = 1; uPrescribed[base + 3] = 0.0;
            isBoundary[base + 4] = 1; uPrescribed[base + 4] = 0.0;
        }

        std::vector<std::size_t> g2f(totalDofs, nposVal);
        std::size_t nfree = 0;
        for (std::size_t i = 0; i < totalDofs; ++i) if (!isBoundary[i]) g2f[i] = nfree++;

        std::vector<std::vector<std::size_t>> elemFreeDofs;
        elemFreeDofs.reserve(numElems);
        for (std::size_t e = 0; e < numElems; ++e)
        {
            std::vector<std::size_t> freeDofs;
            freeDofs.reserve(20);
            for (int a = 0; a < 4; ++a)
            {
                std::size_t node = elems[e][a];
                std::size_t base = node * dofPerNode;
                for (std::size_t d = 0; d < dofPerNode; ++d)
                {
                    std::size_t fIdx = g2f[base + d];
                    if (fIdx != nposVal) freeDofs.push_back(fIdx);
                }
            }
            if (!freeDofs.empty()) elemFreeDofs.push_back(std::move(freeDofs));
        }

        CNum::SparseMatrix<double> Kred(nfree, nfree);
        Kred.PreallocateStructure(elemFreeDofs);
        elemFreeDofs.clear();
        elemFreeDofs.shrink_to_fit();

        std::vector<double> F_red(nfree, 0.0);

        for (std::size_t e = 0; e < numElems; ++e)
        {
            std::size_t nodeI[4];
            for (int a = 0; a < 4; ++a) nodeI[a] = elems[e][a];

            double xCoords[4], yCoords[4];
            for (int a = 0; a < 4; ++a)
            {
                xCoords[a] = nodes[nodeI[a]][0];
                yCoords[a] = nodes[nodeI[a]][1];
            }

            double Ke[20][20] = {};
            double Fe[20] = {};

            for (std::size_t q = 0; q < nqp; ++q)
            {
                const double wq = W[q];
                const std::array<double, 4>& Npt = Nref[q];
                const std::array<double, 4>& dN_dxi_pt = dN_dxi_ref[q];
                const std::array<double, 4>& dN_deta_pt = dN_deta_ref[q];

                double dx_dxi = 0.0, dx_deta = 0.0, dy_dxi = 0.0, dy_deta = 0.0;
                for (int a = 0; a < 4; ++a)
                {
                    dx_dxi += dN_dxi_pt[a] * xCoords[a];
                    dx_deta += dN_deta_pt[a] * xCoords[a];
                    dy_dxi += dN_dxi_pt[a] * yCoords[a];
                    dy_deta += dN_deta_pt[a] * yCoords[a];
                }
                const double detJ = dx_dxi * dy_deta - dx_deta * dy_dxi;
                if (detJ <= std::numeric_limits<double>::epsilon() * 1e4)
                    throw std::runtime_error("Non-positive Jacobian determinant");

                const double invJ00 = dy_deta / detJ;
                const double invJ01 = -dx_deta / detJ;
                const double invJ10 = -dy_dxi / detJ;
                const double invJ11 = dx_dxi / detJ;

                double dN_dx[4], dN_dy[4];
                for (int a = 0; a < 4; ++a)
                {
                    dN_dx[a] = invJ00 * dN_dxi_pt[a] + invJ01 * dN_deta_pt[a];
                    dN_dy[a] = invJ10 * dN_dxi_pt[a] + invJ11 * dN_deta_pt[a];
                }

                const double wdetJ = detJ * wq;

                for (int a = 0; a < 4; ++a)
                {
                    for (int b = 0; b < 4; ++b)
                    {
                        const int rB = a * 5;
                        const int cB = b * 5;

                        double db0u = D_m[0][0] * dN_dx[b] + D_m[0][2] * dN_dy[b];
                        double db1u = D_m[1][0] * dN_dx[b] + D_m[1][2] * dN_dy[b];
                        double db2u = D_m[2][0] * dN_dx[b] + D_m[2][2] * dN_dy[b];
                        double db0v = D_m[0][1] * dN_dy[b] + D_m[0][2] * dN_dx[b];
                        double db1v = D_m[1][1] * dN_dy[b] + D_m[1][2] * dN_dx[b];
                        double db2v = D_m[2][1] * dN_dy[b] + D_m[2][2] * dN_dx[b];

                        Ke[rB + 0][cB + 0] += (dN_dx[a] * db0u + dN_dy[a] * db2u) * wdetJ;
                        Ke[rB + 0][cB + 1] += (dN_dx[a] * db0v + dN_dy[a] * db2v) * wdetJ;
                        Ke[rB + 1][cB + 0] += (dN_dy[a] * db1u + dN_dx[a] * db2u) * wdetJ;
                        Ke[rB + 1][cB + 1] += (dN_dy[a] * db1v + dN_dx[a] * db2v) * wdetJ;

                        double dbb0x = D_b[0][0] * dN_dx[b] + D_b[0][2] * dN_dy[b];
                        double dbb1x = D_b[1][0] * dN_dx[b] + D_b[1][2] * dN_dy[b];
                        double dbb2x = D_b[2][0] * dN_dx[b] + D_b[2][2] * dN_dy[b];
                        double dbb0y = D_b[0][1] * dN_dy[b] + D_b[0][2] * dN_dx[b];
                        double dbb1y = D_b[1][1] * dN_dy[b] + D_b[1][2] * dN_dx[b];
                        double dbb2y = D_b[2][1] * dN_dy[b] + D_b[2][2] * dN_dx[b];

                        Ke[rB + 3][cB + 3] += (dN_dx[a] * dbb0x + dN_dy[a] * dbb2x) * wdetJ;
                        Ke[rB + 3][cB + 4] += (dN_dx[a] * dbb0y + dN_dy[a] * dbb2y) * wdetJ;
                        Ke[rB + 4][cB + 3] += (dN_dy[a] * dbb1x + dN_dx[a] * dbb2x) * wdetJ;
                        Ke[rB + 4][cB + 4] += (dN_dy[a] * dbb1y + dN_dx[a] * dbb2y) * wdetJ;
                    }
                }

                double xgp = 0.0, ygp = 0.0;
                for (int a = 0; a < 4; ++a) { xgp += Npt[a] * xCoords[a]; ygp += Npt[a] * yCoords[a]; }
                const double fq = qfunc(xgp, ygp);
                for (int a = 0; a < 4; ++a)
                    Fe[a * 5 + 2] += Npt[a] * fq * wdetJ;
            }

            {
                const double Nc[4] = { 0.25, 0.25, 0.25, 0.25 };
                const double dxi[4] = { -0.25, 0.25, 0.25, -0.25 };
                const double deta[4] = { -0.25, -0.25, 0.25, 0.25 };

                double dxc = 0, dxe = 0, dyc = 0, dye = 0;
                for (int a = 0; a < 4; ++a)
                {
                    dxc += dxi[a] * xCoords[a]; dxe += deta[a] * xCoords[a];
                    dyc += dxi[a] * yCoords[a]; dye += deta[a] * yCoords[a];
                }
                const double detJc = dxc * dye - dxe * dyc;
                if (detJc <= std::numeric_limits<double>::epsilon() * 1e4)
                    throw std::runtime_error("Non-positive Jacobian at center");

                const double iJ00 = dye / detJc, iJ01 = -dxe / detJc;
                const double iJ10 = -dyc / detJc, iJ11 = dxc / detJc;
                double dNxc[4], dNyc[4];
                for (int a = 0; a < 4; ++a)
                {
                    dNxc[a] = iJ00 * dxi[a] + iJ01 * deta[a];
                    dNyc[a] = iJ10 * dxi[a] + iJ11 * deta[a];
                }

                const double sw = detJc * 4.0;

                for (int a = 0; a < 4; ++a)
                {
                    for (int b = 0; b < 4; ++b)
                    {
                        const int rB = a * 5, cB = b * 5;

                        Ke[rB + 2][cB + 2] += factorS * (dNxc[a] * dNxc[b] + dNyc[a] * dNyc[b]) * sw;

                        Ke[rB + 2][cB + 3] += factorS * (-dNxc[a] * Nc[b]) * sw;
                        Ke[rB + 2][cB + 4] += factorS * (-dNyc[a] * Nc[b]) * sw;

                        Ke[rB + 3][cB + 2] += factorS * (-Nc[a] * dNxc[b]) * sw;
                        Ke[rB + 4][cB + 2] += factorS * (-Nc[a] * dNyc[b]) * sw;

                        Ke[rB + 3][cB + 3] += factorS * (Nc[a] * Nc[b]) * sw;
                        Ke[rB + 4][cB + 4] += factorS * (Nc[a] * Nc[b]) * sw;
                    }
                }
            }

            for (int a = 0; a < 4; ++a)
            {
                std::size_t ArowBase = nodeI[a] * dofPerNode;
                for (int i = 0; i < static_cast<int>(dofPerNode); ++i)
                {
                    std::size_t I = ArowBase + i;
                    std::size_t fI = g2f[I];
                    bool isFreeI = (fI != nposVal);

                    if (isFreeI) F_red[fI] += Fe[a * dofPerNode + i];

                    for (int b = 0; b < 4; ++b)
                    {
                        std::size_t AcolBase = nodeI[b] * dofPerNode;
                        for (int j = 0; j < static_cast<int>(dofPerNode); ++j)
                        {
                            double val = Ke[a * dofPerNode + i][b * dofPerNode + j];
                            if (val == 0.0) continue;

                            std::size_t J = AcolBase + j;
                            std::size_t fJ = g2f[J];
                            bool isFreeJ = (fJ != nposVal);

                            if (isFreeI && isFreeJ) Kred.AddValueFast(fI, fJ, val);
                            else if (isFreeI && !isFreeJ)
                            {
                                double uj = uPrescribed[J];
                                if (uj != 0.0) F_red[fI] -= val * uj;
                            }
                        }
                    }
                }
            }
        }

        Kred.FinalizeFast();

        std::vector<double> diagK;
        Kred.ExtractDiagonal(diagK);

        double minDiag = diagK[0], maxDiag = diagK[0];
        std::size_t negDiagCount = 0;
        for (std::size_t i = 0; i < nfree; ++i)
        {
            if (diagK[i] < minDiag) minDiag = diagK[i];
            if (diagK[i] > maxDiag) maxDiag = diagK[i];
            if (diagK[i] <= 0.0) ++negDiagCount;
        }
        std::cout << "[FEM] nfree=" << nfree << ", diag range=[" << minDiag << ", " << maxDiag
            << "], negDiag=" << negDiagCount << "\n";

        std::vector<double> diagInv(nfree);
        for (std::size_t i = 0; i < nfree; ++i)
        {
            double d = std::abs(diagK[i]);
            diagInv[i] = (d > 1e-30) ? 1.0 / d : 1.0;
        }

        std::vector<double> scale(nfree);
        for (std::size_t i = 0; i < nfree; ++i)
            scale[i] = 1.0 / std::sqrt(std::max(std::abs(diagK[i]), 1e-30));

        for (std::size_t i = 0; i < nfree; ++i)
            F_red[i] *= scale[i];

        {
            auto& vals = const_cast<std::vector<double>&>(Kred.Values());
            const auto& cols = Kred.ColIndex();
            const auto& rptr = Kred.RowPtr();
            for (std::size_t r = 0; r < nfree; ++r)
            {
                for (std::size_t k = rptr[r]; k < rptr[r + 1]; ++k)
                    vals[k] *= scale[r] * scale[cols[k]];
            }
        }

        auto cgProgress = [](std::size_t iter, double resid) {
            if (iter % 500 == 0 || iter <= 5) std::cout << "[FEM-PCG] iter = " << iter << ", resid = " << resid << "\n";
            };

        CNum::Vector<double> xfree = CNum::PCG_RawPointer(
            Kred, F_red.data(), nfree,
            nullptr, nullptr,
            1e-8, 10000, cgProgress);

        for (std::size_t i = 0; i < nfree; ++i)
            xfree[i] *= scale[i];

        std::vector<double> Ufull(totalDofs);
        for (std::size_t g = 0; g < totalDofs; ++g)
        {
            if (isBoundary[g]) Ufull[g] = uPrescribed[g];
            else Ufull[g] = xfree[g2f[g]];
        }

        std::vector<double> outW(numNodes);
        for (std::size_t n = 0; n < numNodes; ++n) outW[n] = Ufull[n * dofPerNode + 2];
        return outW;
    }
} // namespace CNum

int RunFEM2DFromInp(const std::string& inpPath)
{
    try
    {
        std::vector<std::array<double, 2>> nodes;
        std::vector<std::array<std::size_t, 4>> elems;
        std::unordered_map<std::string, std::vector<std::size_t>> nsets;
        double pressure = 0.0, E = 2.1e11, nu = 0.3;

        if (!ParseAbaqusInp(inpPath, nodes, elems, nsets, pressure, E, nu))
        {
            std::cerr << "[RunFEM2DFromInp] Failed to parse INP: " << inpPath << "\n";
            return 3;
        }

        std::cout << "[RunFEM2DFromInp] Parsed " << nodes.size() << " nodes, " << elems.size() << " elements\n";
        std::cout << "[RunFEM2DFromInp] Material E=" << E << ", nu=" << nu << ", pressure=" << pressure << "\n";
        for (const auto& kv : nsets)
            std::cout << "[RunFEM2DFromInp] Nset '" << kv.first << "': " << kv.second.size() << " nodes\n";

        std::vector<std::size_t> boundaryNodes0;
        auto tryAddNset = [&](const char* name) {
            auto it = nsets.find(name);
            if (it != nsets.end())
            {
                for (auto idx0 : it->second)
                {
                    if (idx0 < nodes.size()) boundaryNodes0.push_back(idx0);
                }
                return true;
            }
            return false;
            };

        bool any = false;
        any = tryAddNset("LeftEdge") || any;
        any = tryAddNset("RightEdge") || any;
        any = tryAddNset("TopEdge") || any;
        any = tryAddNset("BottomEdge") || any;

        if (!any)
        {
            std::cout << "[RunFEM2DFromInp] No named nsets found, inferring boundary from coords\n";
            double xmin = nodes[0][0], xmax = nodes[0][0], ymin = nodes[0][1], ymax = nodes[0][1];
            for (auto& n : nodes) { xmin = std::min(xmin, n[0]); xmax = std::max(xmax, n[0]); ymin = std::min(ymin, n[1]); ymax = std::max(ymax, n[1]); }
            const double btol = 1e-9;
            for (std::size_t i = 0; i < nodes.size(); ++i)
            {
                if (std::abs(nodes[i][0] - xmin) <= btol || std::abs(nodes[i][0] - xmax) <= btol ||
                    std::abs(nodes[i][1] - ymin) <= btol || std::abs(nodes[i][1] - ymax) <= btol)
                    boundaryNodes0.push_back(i);
            }
        }

        std::sort(boundaryNodes0.begin(), boundaryNodes0.end());
        boundaryNodes0.erase(std::unique(boundaryNodes0.begin(), boundaryNodes0.end()), boundaryNodes0.end());
        std::cout << "[RunFEM2DFromInp] Boundary nodes: " << boundaryNodes0.size() << "\n";

        auto uniformLoad = [pressure](double, double) { return pressure; };
        const double thickness = 0.01;

        std::vector<double> solW = CNum::SolveMindlinReissner_Unstructured(
            nodes, elems, boundaryNodes0, uniformLoad, E, nu, thickness, 0.0, 1e-10, 2);

        if (solW.empty())
        {
            std::cerr << "[RunFEM2DFromInp] Solver returned empty solution\n";
            return 4;
        }

        double xmin = nodes[0][0], xmax = nodes[0][0], ymin = nodes[0][1], ymax = nodes[0][1];
        for (auto& n : nodes) { xmin = std::min(xmin, n[0]); xmax = std::max(xmax, n[0]); ymin = std::min(ymin, n[1]); ymax = std::max(ymax, n[1]); }
        double xc = 0.5 * (xmin + xmax), yc = 0.5 * (ymin + ymax);
        std::size_t idxCenter = 0;
        double bestDist = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < nodes.size(); ++i)
        {
            double dx = nodes[i][0] - xc, dy = nodes[i][1] - yc;
            double d2 = dx * dx + dy * dy;
            if (d2 < bestDist) { bestDist = d2; idxCenter = i; }
        }

        std::cout << "[RunFEM2DFromInp] Center node index = " << idxCenter << ", w = " << solW[idxCenter] << " m\n";
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Unhandled exception in RunFEM2DFromInp: " << ex.what() << "\n";
        return 2;
    }
}

int RunFEM2D()
{
    const std::string inpPath = "D:\\Code\\C++\\Numericals\\x64\\Debug\\square_plate.inp";
    return RunFEM2DFromInp(inpPath);
}