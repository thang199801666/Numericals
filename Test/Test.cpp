#include "TestFEM2D.h"
#include <fstream>
#include <iomanip>


bool GenerateSquarePlateInp(const std::string& filename,
    double plateSize,
    std::size_t divisions,
    double pressure) noexcept
{
    if (divisions == 0 || plateSize <= 0.0) {
        return false;
    }

    const std::size_t nodesPerEdge = divisions + 1;
    const std::size_t totalNodes = nodesPerEdge * nodesPerEdge;
    const std::size_t totalElems = divisions * divisions;
    const double dx = plateSize / static_cast<double>(divisions);

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        return false;
    }

    ofs << "*Heading\n";
    ofs << "** Generated Abaqus .inp for square plate " << plateSize << " x " << plateSize
        << " with " << divisions << " divisions per edge\n";

    // Nodes
    ofs << "*Node\n";
    ofs << std::fixed << std::setprecision(6);
    std::size_t nodeId = 1;
    for (std::size_t j = 0; j < nodesPerEdge; ++j) {
        double y = static_cast<double>(j) * dx;
        for (std::size_t i = 0; i < nodesPerEdge; ++i) {
            double x = static_cast<double>(i) * dx;
            ofs << nodeId << ", " << x << ", " << y << ", 0.0\n";
            ++nodeId;
        }
    }

    // Elements (4-node quads). Node numbering: lower-left, lower-right, upper-right, upper-left (CCW)
    ofs << "*Element, type=CPS4\n";
    std::size_t elemId = 1;
    for (std::size_t j = 0; j < divisions; ++j) {
        for (std::size_t i = 0; i < divisions; ++i) {
            // node(i,j) = j*nodesPerEdge + i + 1
            const std::size_t n1 = j * nodesPerEdge + i + 1;               // lower-left
            const std::size_t n2 = n1 + 1;                                // lower-right
            const std::size_t n3 = n2 + nodesPerEdge;                     // upper-right
            const std::size_t n4 = n1 + nodesPerEdge;                     // upper-left
            ofs << elemId << ", " << n1 << ", " << n2 << ", " << n3 << ", " << n4 << "\n";
            ++elemId;
        }
    }

    // Element set and node set (all)
    ofs << "*Elset, elset=AllElements, generate\n";
    ofs << 1 << ", " << (totalElems) << ", 1\n";
    ofs << "*Nset, nset=AllNodes, generate\n";
    ofs << 1 << ", " << (totalNodes) << ", 1\n";

    // Node sets for edges (use generate with arithmetic sequences)
    // Left edge: nodes 1, 1+nodesPerEdge, ..., 1+(nodesPerEdge-1)*nodesPerEdge
    const std::size_t leftStart = 1;
    const std::size_t leftEnd = 1 + (nodesPerEdge - 1) * nodesPerEdge;
    const std::size_t rightStart = nodesPerEdge;
    const std::size_t rightEnd = nodesPerEdge * nodesPerEdge;
    const std::size_t bottomStart = 1;
    const std::size_t bottomEnd = nodesPerEdge;
    const std::size_t topStart = (nodesPerEdge - 1) * nodesPerEdge + 1;
    const std::size_t topEnd = nodesPerEdge * nodesPerEdge;

    ofs << "*Nset, nset=LeftEdge, generate\n";
    ofs << leftStart << ", " << leftEnd << ", " << nodesPerEdge << "\n";
    ofs << "*Nset, nset=RightEdge, generate\n";
    ofs << rightStart << ", " << rightEnd << ", " << nodesPerEdge << "\n";
    ofs << "*Nset, nset=BottomEdge, generate\n";
    ofs << bottomStart << ", " << bottomEnd << ", 1\n";
    ofs << "*Nset, nset=TopEdge, generate\n";
    ofs << topStart << ", " << topEnd << ", 1\n";

    // Boundaries: fully fix both DOFs on all four edges
    ofs << "*Boundary\n";
    ofs << "LeftEdge, 1, 1, 0.0\n";
    ofs << "LeftEdge, 2, 2, 0.0\n";
    ofs << "RightEdge, 1, 1, 0.0\n";
    ofs << "RightEdge, 2, 2, 0.0\n";
    ofs << "BottomEdge, 1, 1, 0.0\n";
    ofs << "BottomEdge, 2, 2, 0.0\n";
    ofs << "TopEdge, 1, 1, 0.0\n";
    ofs << "TopEdge, 2, 2, 0.0\n";

    // Material (Steel)
    ofs << "*Material, name=Steel\n";
    ofs << "*Elastic\n";
    ofs << 210000000000.0 << ", " << 0.3 << "\n"; // E (Pa), Nu

    // Simple step with static analysis and a distributed surface pressure on all elements
    ofs << "*Step, name=LoadStep, nlgeom=NO\n";
    ofs << "*Static\n";
    ofs << "1., 1., 1e-05, 1.\n";
    ofs << "*Dload, elset=AllElements\n";
    ofs << "P, " << pressure << "\n";
    ofs << "*End Step\n";

    ofs.close();
    return true;
}

int main()
{
    return RunFEM2D();
}