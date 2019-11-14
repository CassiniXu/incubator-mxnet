#include "./np_diagflat_op-inl.h"

namespace mxnet{
namespace op{

DMLC_REGISTER_PARAMETER(NumpyDiagflatParam);

NNVM_REGISTER_OP(_npi_diagflat)
.describe(R"code(contructs a diagonal array.
``diagflat``'s behavioir is independent form input array dimensions;

- N-D arrays: constructs a 2-D array with the input as its diagonal, all other elements a zeroes. 

Examples::
	x = [[1,2,3]]

	diagflat(x) = [[1,0,0],
					[0,2,0],
					[0,0,3]]

	diagflat(x,k=1) = [[0,1,0,0],
						[[0,0,2,0],
						[[0,0,0,3]]]

	diagflat(x,k=-1) = [[0,0,0,0],
						[1,0,0,0],
						[0,2,0,0],
						[0,0,3,0]]



	x = [[1,2],[3,4]]

	diagflat(x) = [[1,0,0,0],
					[0,2,0,0],
					[0,0,3,0],
					[0,0,0,4]]



	x = [[[1,2],[3,4]],[[5,6],[7,8]]]

	diagflat(x) = [[1,0,0,0,0,0,0,0],
					[0,2,0,0,0,0,0,0],
					[0,0,3,0,0,0,0,0],
					[0,0,0,4,0,0,0,0],
					[0,0,0,0,5,0,0,0],
					[0,0,0,0,0,6,0,0],
					[0,0,0,0,0,0,7,0],
					[0,0,0,0,0,0,0,8]]
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<NumpyDiagflatParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
	[](const NodeAttrs& attrs) {
		return std::vector<std::string>{"data"};
	})
.set_attr<mxnet::FInferShape>("FInferShape", NumpyDiagflatOpShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyDiagflatOpType)
.set_attr<FCompute>("FCompute<cpu>",NumpyDiagflatOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",ElemwiseGradUseNone{"_backward_npi_diagflat"})
.add_argument("data","NDArray-or-Symbol","Input ndarray")
.add_arguments(NumpyDiagflatParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_npi_diagflat)
.set_attr_parser(ParamParser<NumpyDiagflatParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward",true)
.set_attr<FCompute>("FCompute<cpu>",NumpyDiagflatOpBackward<cpu>);


} // namespace no
} // namespace mxnet