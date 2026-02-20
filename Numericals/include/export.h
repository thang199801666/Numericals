#pragma once

// Simple cross-platform export macro for Windows DLLs.
// Define BUILDING_NUMERICALS in the Numericals DLL project.
#if defined(_WIN32) || defined(_WIN64)
#if defined(BUILDING_NUMERICALS)
#define NUMERICALS_API __declspec(dllexport)
#else
#define NUMERICALS_API __declspec(dllimport)
#endif
#else
#define NUMERICALS_API
#endif