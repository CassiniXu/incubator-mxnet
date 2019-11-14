#ifndef MXNET_OPERATOR_NUMPY_DIAGFLAT_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_DIAGFLAT_OP_INL_H_

#include <dmlc/parameter.h>
#include <vector>
#include <algorithm>
#include <utility>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct NumpyDiagflatParam : public dmlc::Parameter<NumpyDiagflatParam> {
    int k;
    DMLC_DECLARE_PARAMETER(NumpyDiagflatParam) {
        DMLC_DECLARE_FIELD(k)
            .set_default(0)
            .describe("Diagonal in question. The default is 0. "
                "Use k>0 for diagonals above the main diagonal, "
                "and k<0 for diagonals below the main diagonal. "
            );
    }
};



inline mxnet::TShape NumpyDiagflatShapeImpl(const mxnet::TShape& ishape, const int k) 
{
    if (ishape.ndim() == 1) {
        auto s = ishape[0] + std::abs(k);
        return mxnet::TShape({s, s});
    }

    if (ishape.ndim() >=2 ){
    	auto s = 1;
    	for(uint32_t i = 0; i < ishape.ndim(); i++){
    		if(ishape[i] >= 2){
    			s = s * ishape[i];
    		}
    	}
    	s = s + std::abs(k);
    	return mxnet::TShape({s,s});
    }
}

inline bool NumpyDiagflatOpShape(const nnvm::NodeAttrs& attrs,
                        mxnet::ShapeVector* in_attrs,
                        mxnet::ShapeVector* out_attrs) {
    CHECK_EQ(in_attrs->size(), 1U);
    CHECK_EQ(out_attrs->size(), 1U);

    const mxnet::TShape& ishape = (*in_attrs)[0];

    if (!mxnet::ndim_is_known(ishape)) {
        return false;
    }

    const NumpyDiagflatParam& param = nnvm::get<NumpyDiagflatParam>(attrs.parsed);

    mxnet::TShape oshape = NumpyDiagflatShapeImpl(ishape,
                                         param.k);

    if (shape_is_none(oshape)) {
        LOG(FATAL) << "Diagonal does not exist.";
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);

    return shape_is_known(out_attrs->at(0));
}



inline bool NumpyDiagflatOpType(const nnvm::NodeAttrs& attrs,
                       std::vector<int> *in_attrs,
                       std::vector<int> *out_attrs) {
    CHECK_EQ(in_attrs->size(), 1U);
    CHECK_EQ(out_attrs->size(), 1U);

    TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
    TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
    return (*out_attrs)[0] != -1;
}



template<int req, bool back>
struct diagflat_gen {
    template<typename DType>
    MSHADOW_XINLINE static void Map(index_t i, 
                                    DType* out, 
                                    const DType* a,
                                    mshadow::Shape<2> oshape,
                                    int k){
        using namespace mxnet_op;
        auto j = unravel(i,oshape);
        if (j[1] == j[0] + k){
            auto l = j[0] < j[1] ? j[0] : j[1];
            if (back){
                KERNEL_ASSIGN(out[l],req,a[i]);
            }else{
                KERNEL_ASSIGN(out[i],req,a[l]);
            }
        }else if(!back){
            KERNEL_ASSIGN(out[i],req,static_cast<DType>(0));
        }
    }
};





template<typename xpu, bool back>
void NumpyDiagflatOpProcess(const TBlob& in_data,
                           const TBlob& out_data,
                           const mxnet::TShape& ishape,
                           const mxnet::TShape& oshape,
                           index_t dsize,
                           const NumpyDiagflatParam& param,
                           mxnet_op::Stream<xpu> *s,
                           const std::vector<OpReqType>& req) {

    using namespace mxnet_op;
    using namespace mshadow;
    MSHADOW_TYPE_SWITCH(out_data.type_flag_,DType,{
        MXNET_ASSIGN_REQ_SWITCH(req[0],req_type,{
            Kernel<diagflat_gen<req_type,back>, xpu>::Launch(s,
                                                    dsize,
                                                    out_data.dptr<DType>(),
                                                    in_data.dptr<DType>(),
                                                    Shape2(oshape[0],oshape[1]),
                                                    param.k);
        });
    });

}


template<typename xpu>
void NumpyDiagflatOpForward(const nnvm::NodeAttrs& attrs,
							const OpContext& ctx,
							const std::vector<TBlob>& inputs,
							const std::vector<OpReqType>& req,
							const std::vector<TBlob>& outputs)
{
	using namespace mxnet_op;
	using namespace	mshadow;
	CHECK_EQ(inputs.size(), 1U);
	CHECK_EQ(outputs.size(), 1U);
	CHECK_EQ(req.size(), 1U);
    CHECK_EQ(req[0], kWriteTo);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const TBlob& in_data = inputs[0];
    const TBlob& out_data = outputs[0];
    const mxnet::TShape& ishape = inputs[0].shape_;
    const mxnet::TShape& oshape = outputs[0].shape_;
    const NumpyDiagflatParam& param = nnvm::get<NumpyDiagflatParam>(attrs.parsed);
    NumpyDiagflatOpProcess<xpu, false>(in_data, out_data, ishape, oshape, out_data.Size(), param, s, req);
}


template<typename xpu>
void NumpyDiagflatOpBackward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
    using namespace mxnet_op;
    using namespace mshadow;
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    const TBlob& in_data = inputs[0];
    const TBlob& out_data = outputs[0];
    const mxnet::TShape& ishape = inputs[0].shape_;
    const mxnet::TShape& oshape = outputs[0].shape_;
    const NumpyDiagflatParam& param = nnvm::get<NumpyDiagflatParam>(attrs.parsed);

    NumpyDiagflatOpProcess<xpu, true>(in_data, out_data, oshape, ishape, in_data.Size(), param, s, req);
}


}
}

#endif