#include "include/CArray.h"
#include "include/ufuncs.h"
#include "include/vector.h"
#include "include/matrix.h"
#include "include/linalg.h"

// Explicit instantiations emitted into the DLL.
// Note: the Matrix template is annotated with NUMERICALS_API in the header,
// so these plain explicit instantiations will produce exported symbols.
template class CNum::Matrix<double>;
template class CNum::Matrix<int>;
template class CNum::Vector<double>;
template class CNum::Vector<std::size_t>;