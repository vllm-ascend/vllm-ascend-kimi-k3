/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/op_def_registry.h"

#include <initializer_list>

namespace ops {
class ChunkKdaBwdIntra : public OpDef {
public:
    explicit ChunkKdaBwdIntra(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> bf16 = {ge::DT_BF16};
        const std::initializer_list<ge::DataType> fp32 = {ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> int64 = {ge::DT_INT64};
        const std::initializer_list<ge::Format> ndFormat = {ge::FORMAT_ND};
        this->Input("q").ParamType(REQUIRED).DataType(bf16).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Input("k").ParamType(REQUIRED).DataType(bf16).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Input("gk").ParamType(REQUIRED).DataType(fp32).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Input("beta").ParamType(REQUIRED).DataType(bf16).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Input("dAqk").ParamType(REQUIRED).DataType(fp32).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Input("dAkk").ParamType(REQUIRED).DataType(fp32).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Input("dq").ParamType(REQUIRED).DataType(fp32).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Input("dk").ParamType(REQUIRED).DataType(fp32).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Input("db").ParamType(REQUIRED).DataType(fp32).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Input("dg").ParamType(REQUIRED).DataType(fp32).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(int64).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Input("chunk_metadata").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(int64).Format(ndFormat).UnknownShapeFormat(ndFormat);

        this->Output("dq_out").ParamType(REQUIRED).DataType(fp32).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Output("dk_out").ParamType(REQUIRED).DataType(fp32).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Output("db_out").ParamType(REQUIRED).DataType(fp32).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Output("dg_out").ParamType(REQUIRED).DataType(fp32).Format(ndFormat).UnknownShapeFormat(ndFormat);

        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("safe_gate").AttrType(REQUIRED).Bool(true);
        this->Attr("layout_mode").AttrType(REQUIRED).Int(0);
        this->Attr("total_chunks").AttrType(REQUIRED).Int(0);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("opFile.value", "chunk_kda_bwd_intra")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(ChunkKdaBwdIntra);
} // namespace ops
