#include <iostream>
#include <cmath>
#include <exception>
#include <vector>
#include "NonlinearEquations.h"

using Numericals::NonlinearEquations;
using Numericals::RootResult;

static bool approx_eq(double a, double b, double eps = 1e-9)
{
    return std::abs(a - b) <= eps * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

int main()
{
    try
    {
        int failures = 0;
        auto check = [&](bool cond, const char* name) {
            if (cond)
                std::cout << "[PASS] " << name << "\n";
            else
            {
                std::cout << "[FAIL] " << name << "\n";
                ++failures;
            }
            };

        // 1) LinearIncrementalMethod (detect exact zero on grid)
        {
            auto f = [](double x) { return x - 1.0; };
            std::vector<double> candidates = NonlinearEquations::LinearIncrementalMethod(f, 0.0, 2.0, 0.5);
            bool found = false;
            for (double c : candidates)
            {
                if (approx_eq(c, 1.0, 1e-12))
                {
                    found = true;
                    break;
                }
            }
            check(found, "LinearIncrementalMethod finds x == 1.0 on grid");
        }

        // Prepare common problem f(x) = x^2 - 2 (root at sqrt(2))
        auto f = [](double x) { return x * x - 2.0; };
        auto df = [](double x) { return 2.0 * x; };
        const double sqrt2 = std::sqrt(2.0);

        // 2) BisectionMethod
        {
            RootResult r = NonlinearEquations::BisectionMethod(f, 0.0, 2.0, 1e-12, 100);
            bool ok = r.converged && approx_eq(r.root, sqrt2, 1e-12);
            check(ok, "BisectionMethod finds sqrt(2)");
        }

        // 3) TheSecantMethod
        {
            RootResult r = NonlinearEquations::TheSecantMethod(f, 0.0, 2.0, 1e-12, 200);
            bool ok = r.converged && approx_eq(r.root, sqrt2, 1e-9);
            check(ok, "TheSecantMethod finds sqrt(2)");
        }

        // 4) FalsePositioningMethod (Regula Falsi)
        {
            RootResult r = NonlinearEquations::FalsePositioningMethod(f, 0.0, 2.0, 1e-12, 200);
            bool ok = r.converged && approx_eq(r.root, sqrt2, 1e-9);
            check(ok, "FalsePositioningMethod finds sqrt(2)");
        }

        // 5) FixedPointIteration using the Babylonian iteration g(x) = 0.5*(x + 2/x)
        {
            auto g = [](double x) { return 0.5 * (x + 2.0 / x); };
            RootResult r = NonlinearEquations::FixedPointIteration(g, 1.0, 1e-12, 1000);
            bool ok = r.converged && approx_eq(r.root, sqrt2, 1e-12);
            check(ok, "FixedPointIteration (Babylonian) converges to sqrt(2)");
        }

        // 6) NewtonRaphsonMethod
        {
            RootResult r = NonlinearEquations::NewtonRaphsonMethod(f, df, 1.0, 1e-12, 100);
            bool ok = r.converged && approx_eq(r.root, sqrt2, 1e-12);
            check(ok, "NewtonRaphsonMethod finds sqrt(2)");
        }

        // Summary
        if (failures == 0)
        {
            std::cout << "\nAll NonlinearEquations tests passed.\n";
            return 0;
        }
        else
        {
            std::cout << "\n" << failures << " NonlinearEquations test(s) failed.\n";
            return 1;
        }
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Unhandled exception: " << ex.what() << "\n";
        return 2;
    }
}