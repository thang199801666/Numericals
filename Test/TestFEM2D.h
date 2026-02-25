#pragma once

#include <string>
#include <vector>
#include <array>
#include <unordered_map>

int RunFEM2D();
int RunFEM2DFromInp(const std::string& inpPath);

bool ParseAbaqusInp(const std::string& filename,
    std::vector<std::array<double, 2>>& nodes,
    std::vector<std::array<std::size_t, 4>>& elems,
    std::unordered_map<std::string, std::vector<std::size_t>>& nsets,
    double& outPressure,
    double& outE,
    double& outNu) noexcept;