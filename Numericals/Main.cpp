#include "Numericals/ndarray.h"
#include "Numericals/ufuncs.h"
#include "Numericals/vector.h"
#include "Numericals/matrix.h"
#include "Numericals/linalg.h"

#include <iostream>
#include <cmath>

static int g_tests_run = 0;
static int g_tests_failed = 0;

#define EXPECT_TRUE(cond) \
    do { ++g_tests_run; bool _val = static_cast<bool>(cond); \
         if (!_val) { std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__ \
                              << " EXPECT_TRUE(" << #cond << ") evaluated to false\n"; ++g_tests_failed; } \
    } while(0)

#define EXPECT_EQ_INT(a,b) \
    do { ++g_tests_run; auto _a = (a); auto _b = (b); \
         if (!(_a == _b)) { std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__ \
                                     << " " << #a << " != " << #b << " (" << _a << " vs " << _b << ")\n"; ++g_tests_failed; } \
    } while(0)

#define EXPECT_NEAR(a,b,eps) \
    do { ++g_tests_run; double _ea = static_cast<double>((a)); double _eb = static_cast<double>((b)); double _ee = static_cast<double>((eps)); \
         double _diff = std::fabs(_ea - _eb); \
         if (_diff > _ee) { std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__ \
                                   << " EXPECT_NEAR(" << #a << "," << #b << "," << #eps << ") -> (" \
                                   << _ea << " vs " << _eb << "), diff=" << _diff << "\n"; ++g_tests_failed; } \
    } while(0)

#define EXPECT_THROW(expr) \
    do { ++g_tests_run; bool _ok=false; try { expr; } catch (const std::exception& _e) { _ok=true; } catch(...) { _ok=true; } \
         if (!_ok) { std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__ << " EXPECT_THROW(" << #expr << ") did not throw\n"; ++g_tests_failed; } \
    } while(0)

using CNum::NdArray;
using CNum::shape_t;
using CNum::Vector;
using CNum::Matrix;

static bool shapes_equal(const shape_t& a, const shape_t& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}

void test_matrix_features()
{
    // construct 2x3 and set elements via operator()
    Matrix<double> M(2, 3);
    M(0, 0) = 1.0; M(0, 1) = 2.0; M(0, 2) = 3.0;
    M(1, 0) = 4.0; M(1, 1) = 5.0; M(1, 2) = 6.0;
    EXPECT_EQ_INT(M.rows(), 2);
    EXPECT_EQ_INT(M.cols(), 3);
    EXPECT_NEAR(M(0, 2), 3.0, 1e-12);

    // operator[] row proxy access
    M[0][1] = 7.5;
    EXPECT_NEAR(M(0, 1), 7.5, 1e-12);

    // assign whole row from Vector and initializer_list
    M.row(0) = Vector<double>{ 1.0, 2.0, 3.0 };
    EXPECT_NEAR(M(0, 0), 1.0, 1e-12);
    M[1] = std::initializer_list<double>{ 9.0, 8.0, 7.0 };
    EXPECT_NEAR(M(1, 2), 7.0, 1e-12);

    // assign whole column from Vector
    Matrix<double> N(3, 2, 0.0);
    N.col(0) = Vector<double>{ 10.0, 11.0, 12.0 };
    EXPECT_NEAR(N(2, 0), 12.0, 1e-12);

    // transpose and .T
    Matrix<double> At = M.transpose();
    Matrix<double> At2 = M.T;
    EXPECT_TRUE(shapes_equal(At.to_ndarray().shape(), shape_t{ 3, 2 }));
    EXPECT_NEAR(At(1, 0), M(0, 1), 1e-12);
    EXPECT_NEAR(At2(1, 0), M(0, 1), 1e-12);

    // trace and determinant
    Matrix<double> S{ {1.0, 2.0}, {3.0, 4.0} }; // 2x2
    EXPECT_NEAR(S.trace(), 5.0, 1e-12);
    EXPECT_NEAR(S.det(), -2.0, 1e-12);

    Matrix<double> I3{ {1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0} };
    EXPECT_NEAR(I3.det(), 1.0, 1e-12);

    // det on non-square must throw
    Matrix<double> nonsq(2, 3);
    EXPECT_THROW(nonsq.det());

    // matrix * vector (matmul) and operator*
    Vector<double> x{ 5.0, 6.0 };
    Matrix<double> A{ {1.0,2.0},{3.0,4.0} }; // 2x2
    Vector<double> y1 = CNum::matmul(A, x);
    Vector<double> y2 = A * x;
    EXPECT_NEAR(y1[0], y2[0], 1e-12);
    EXPECT_NEAR(y1[1], y2[1], 1e-12);

    // matrix * matrix (matmul) and hadamard
    Matrix<double> B{ {2.0,0.0},{1.0,2.0} };
    Matrix<double> C1 = CNum::matmul(A, B);
    Matrix<double> C2 = A * B;
    EXPECT_NEAR(C1(0, 0), C2(0, 0), 1e-12);
    EXPECT_NEAR(C1(1, 1), C2(1, 1), 1e-12);

    Matrix<double> H = CNum::hadamard(A, B);
    EXPECT_NEAR(H(1, 1), A(1, 1) * B(1, 1), 1e-12);

    // scalar ops
    Matrix<double> m_plus = A + 1.0;
    EXPECT_NEAR(m_plus(0, 0), 2.0, 1e-12);
    Matrix<double> m_scaled = 2.0 * A;
    EXPECT_NEAR(m_scaled(1, 0), 6.0, 1e-12);

    // mismatched row assignment should throw
    Vector<double> shortRow{ 1.0, 2.0 }; // create temporary outside macro to avoid comma splitting
    EXPECT_THROW(M.row(0) = shortRow); // M has 3 columns

    // to_ndarray round-trip basic check
    auto arr = A.to_ndarray();
    EXPECT_EQ_INT(static_cast<int>(arr.size()), static_cast<int>(A.rows() * A.cols()));
}

int main()
{
    std::cout << "Running Numericals unit tests...\n";

    test_matrix_features();

    std::cout << "Tests run: " << g_tests_run << ", Failures: " << g_tests_failed << "\n";
    return (g_tests_failed == 0) ? 0 : 1;
}