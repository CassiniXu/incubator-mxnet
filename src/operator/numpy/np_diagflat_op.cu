#include "./np_dagflat_op-inl.h"

namespace mxnet{
namespace op{

NNVM_REGISTER_OP(diagflat)
.set_attr<FCompute>("FComputr<gpu>", NumpyDiagflatOpForward<gpu>);

NNVM_REGISTER_OP(_backwad_diagflat_)
.set_attr<FCompute>("FCompute<gpu>", NumpyDiagflatOpBackward<gpu>);

} // namespace op
} // namespace mxnet