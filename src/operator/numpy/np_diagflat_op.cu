#include "./np_diagflat_op-inl.h"

namespace mxnet{
namespace op{

NNVM_REGISTER_OP(_npi_diagflat)
.set_attr<FCompute>("FCompute<gpu>", NumpyDiagflatOpForward<gpu>);

NNVM_REGISTER_OP(_backward_npi_diagflat)
.set_attr<FCompute>("FCompute<gpu>", NumpyDiagflatOpBackward<gpu>);

} // namespace op
} // namespace mxnet