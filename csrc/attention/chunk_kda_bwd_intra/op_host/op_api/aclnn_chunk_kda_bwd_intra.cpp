/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_chunk_kda_bwd_intra.h"
#include "chunk_kda_bwd_intra.h"
#include "../../../kda_layout_swap12/op_host/op_api/kda_layout_swap12.h"

#include <cstring>
#include <initializer_list>
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {

struct Params {
    const aclTensor *q;
    const aclTensor *k;
    const aclTensor *gk;
    const aclTensor *beta;
    const aclTensor *dAqk;
    const aclTensor *dAkk;
    const aclTensor *dq;
    const aclTensor *dk;
    const aclTensor *db;
    const aclTensor *dg;
    const aclIntArray *cuSeqlens;
    const aclIntArray *chunkIndices;
    int64_t chunkSize;
    bool safeGate;
    const char *layout;
    const aclTensor *dqOut;
    const aclTensor *dkOut;
    const aclTensor *dbOut;
    const aclTensor *dgOut;
};

enum class Layout {
    BSND,
    BNSD,
    TND,
};

op::Shape MakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

bool SameShape(const aclTensor *a, const aclTensor *b)
{
    const auto lhs = a->GetViewShape();
    const auto rhs = b->GetViewShape();
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

aclnnStatus ParseLayout(const char *text, Layout &layout)
{
    CHECK_COND(text != nullptr, ACLNN_ERR_PARAM_INVALID,
               "layout must not be nullptr and must be BSND, BNSD or TND.");
    if (std::strcmp(text, "BSND") == 0) {
        layout = Layout::BSND;
        return ACLNN_SUCCESS;
    }
    if (std::strcmp(text, "BNSD") == 0) {
        layout = Layout::BNSD;
        return ACLNN_SUCCESS;
    }
    if (std::strcmp(text, "TND") == 0) {
        layout = Layout::TND;
        return ACLNN_SUCCESS;
    }
    CHECK_COND(false, ACLNN_ERR_PARAM_INVALID,
               "ChunkKdaBwdIntra supports dense BSND/BNSD or varlen TND.");
}

aclnnStatus ValidateVarLenMetadata(
    const Params &p, int64_t totalTokens, int64_t &totalChunks)
{
    CHECK_COND(p.cuSeqlens != nullptr, ACLNN_ERR_PARAM_INVALID,
               "varlen input requires cu_seqlens.");
    CHECK_COND(p.cuSeqlens->Size() >= 2 && p.cuSeqlens->Size() <= 65,
               ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens must contain 2..65 entries.");
    const aclIntArray &cu = *p.cuSeqlens;
    CHECK_COND(cu[0] == 0, ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens[0] must be zero.");
    CHECK_COND(cu[cu.Size() - 1] == totalTokens, ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens[-1] must equal total T.");

    totalChunks = 0;
    for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
        CHECK_COND(cu[seq] >= 0 && cu[seq + 1] >= cu[seq],
                   ACLNN_ERR_PARAM_INVALID,
                   "cu_seqlens must be nondecreasing.");
        const int64_t length = cu[seq + 1] - cu[seq];
        totalChunks += (length + p.chunkSize - 1) / p.chunkSize;
    }
    CHECK_COND(totalChunks > 0, ACLNN_ERR_PARAM_INVALID,
               "varlen input must contain at least one non-empty sequence.");

    if (p.chunkIndices != nullptr) {
        CHECK_COND(
            p.chunkIndices->Size() == static_cast<size_t>(totalChunks) * 2,
            ACLNN_ERR_PARAM_INVALID,
            "chunk_indices must contain exactly two values per chunk.");
        size_t index = 0;
        for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
            const int64_t length = cu[seq + 1] - cu[seq];
            const int64_t count =
                (length + p.chunkSize - 1) / p.chunkSize;
            for (int64_t localChunk = 0; localChunk < count; ++localChunk) {
                CHECK_COND(
                    (*p.chunkIndices)[index] == static_cast<int64_t>(seq) &&
                        (*p.chunkIndices)[index + 1] == localChunk,
                    ACLNN_ERR_PARAM_INVALID,
                    "chunk_indices must use canonical sequence-major order.");
                index += 2;
            }
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Check(const Params &p, Layout &layout, int64_t &totalChunks)
{
    const aclTensor *required[] = {
        p.q, p.k, p.gk, p.beta, p.dAqk, p.dAkk, p.dq, p.dk, p.db, p.dg,
        p.dqOut, p.dkOut, p.dbOut, p.dgOut
    };
    for (const aclTensor *tensor : required) {
        CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "ChunkKdaBwdIntra tensor arguments must not be nullptr.");
        CHECK_COND(IsContiguous(tensor), ACLNN_ERR_PARAM_INVALID,
                   "ChunkKdaBwdIntra only supports contiguous tensors.");
    }
    const aclTensor *bf16[] = {p.q, p.k, p.beta};
    for (const aclTensor *tensor : bf16) {
        CHECK_COND(tensor->GetDataType() == DataType::DT_BF16,
                   ACLNN_ERR_PARAM_INVALID, "q/k/beta must be BF16.");
    }
    const aclTensor *fp32[] = {
        p.gk, p.dAqk, p.dAkk, p.dq, p.dk, p.db, p.dg,
        p.dqOut, p.dkOut, p.dbOut, p.dgOut
    };
    for (const aclTensor *tensor : fp32) {
        CHECK_COND(tensor->GetDataType() == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID,
                   "gk/dA/gradient inputs and outputs must be FP32.");
    }
    CHECK_RET(ParseLayout(p.layout, layout) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(p.safeGate, ACLNN_ERR_PARAM_INVALID,
               "safe_gate=false is reserved but not supported in v1.");
    CHECK_COND(p.chunkSize == 64, ACLNN_ERR_PARAM_INVALID,
               "chunk_size must be 64.");
    const bool isVarLen = p.cuSeqlens != nullptr;
    CHECK_COND(isVarLen || p.chunkIndices == nullptr,
               ACLNN_ERR_PARAM_INVALID,
               "chunk_indices requires cu_seqlens.");
    CHECK_COND((isVarLen && layout != Layout::BNSD) ||
                   (!isVarLen && layout != Layout::TND),
               ACLNN_ERR_PARAM_INVALID,
               "varlen supports TND/BSND; dense supports BSND/BNSD.");
    const auto q = p.q->GetViewShape();
    const size_t expectedRank = layout == Layout::TND ? 3 : 4;
    CHECK_COND(q.GetDimNum() == expectedRank, ACLNN_ERR_PARAM_INVALID,
               "q rank does not match layout.");
    const int64_t b = layout == Layout::TND ? 1 : q.GetDim(0);
    const int64_t t = layout == Layout::TND ? q.GetDim(0) :
                      q.GetDim(layout == Layout::BSND ? 1 : 2);
    const int64_t h = layout == Layout::TND ? q.GetDim(1) :
                      q.GetDim(layout == Layout::BSND ? 2 : 1);
    const int64_t k = layout == Layout::TND ? q.GetDim(2) : q.GetDim(3);
    CHECK_COND(b > 0 && h > 0 && t > 0, ACLNN_ERR_PARAM_INVALID,
               "B/H/T must be positive.");
    CHECK_COND((isVarLen && k == 128) ||
                   (!isVarLen && (k == 64 || k == 128 || k == 256)),
               ACLNN_ERR_PARAM_INVALID,
               "varlen supports K=128; dense supports K=64/128/256.");
    CHECK_COND(!isVarLen || layout != Layout::BSND || b == 1,
               ACLNN_ERR_PARAM_INVALID,
               "varlen BSND compatibility requires B=1.");
    CHECK_COND(SameShape(p.q, p.k) && SameShape(p.q, p.gk) &&
                   SameShape(p.q, p.dq) && SameShape(p.q, p.dk) &&
                   SameShape(p.q, p.dg) && SameShape(p.q, p.dqOut) &&
                   SameShape(p.q, p.dkOut) && SameShape(p.q, p.dgOut),
               ACLNN_ERR_PARAM_INVALID,
               "q/k/gk/dq/dk/dg and vector outputs must have identical shape.");
    const auto beta = p.beta->GetViewShape();
    const size_t scalarRank = layout == Layout::TND ? 2 : 3;
    CHECK_COND(beta.GetDimNum() == scalarRank, ACLNN_ERR_PARAM_INVALID,
               "beta/db/dbOut rank does not match layout.");
    const bool scalarDimsMatch = layout == Layout::TND ?
        (beta.GetDim(0) == t && beta.GetDim(1) == h) :
        (beta.GetDim(0) == b &&
         beta.GetDim(layout == Layout::BSND ? 1 : 2) == t &&
         beta.GetDim(layout == Layout::BSND ? 2 : 1) == h);
    CHECK_COND(scalarDimsMatch && SameShape(p.beta, p.db) &&
                   SameShape(p.beta, p.dbOut),
               ACLNN_ERR_PARAM_INVALID,
               "beta/db/dbOut shape does not match layout.");
    const auto da = p.dAqk->GetViewShape();
    const size_t matrixRank = layout == Layout::TND ? 3 : 4;
    CHECK_COND(da.GetDimNum() == matrixRank, ACLNN_ERR_PARAM_INVALID,
               "dAqk/dAkk rank does not match layout.");
    const bool matrixDimsMatch = layout == Layout::TND ?
        (da.GetDim(0) == t && da.GetDim(1) == h &&
         da.GetDim(2) == p.chunkSize) :
        (da.GetDim(0) == b &&
         da.GetDim(layout == Layout::BSND ? 1 : 2) == t &&
         da.GetDim(layout == Layout::BSND ? 2 : 1) == h &&
         da.GetDim(3) == p.chunkSize);
    CHECK_COND(matrixDimsMatch && SameShape(p.dAqk, p.dAkk),
               ACLNN_ERR_PARAM_INVALID,
               "dAqk/dAkk shape does not match layout.");
    totalChunks = 0;
    if (isVarLen) {
        CHECK_RET(
            ValidateVarLenMetadata(p, t, totalChunks) == ACLNN_SUCCESS,
            ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

} // namespace

extern "C" aclnnStatus aclnnChunkKdaBwdIntraGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *gk, const aclTensor *beta,
    const aclTensor *dAqk, const aclTensor *dAkk, const aclTensor *dq, const aclTensor *dk,
    const aclTensor *db, const aclTensor *dg, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, int64_t chunkSize, bool safeGate,
    const char *layout, const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dbOut, const aclTensor *dgOut, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkKdaBwdIntra,
                   DFX_IN(q, k, gk, beta, dAqk, dAkk, dq, dk, db, dg,
                          cuSeqlensOptional, chunkIndicesOptional, chunkSize, safeGate, layout),
                   DFX_OUT(dqOut, dkOut, dbOut, dgOut));
    CHECK_COND(workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "workspaceSize and executor must not be nullptr.");
    Params params{q, k, gk, beta, dAqk, dAkk, dq, dk, db, dg,
                  cuSeqlensOptional, chunkIndicesOptional, chunkSize, safeGate, layout,
                  dqOut, dkOut, dbOut, dgOut};
    Layout parsedLayout;
    int64_t totalChunks = 0;
    CHECK_RET(Check(params, parsedLayout, totalChunks) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    aclOpExecutor *executorPtr = uniqueExecutor.get();

    const bool isVarLen = cuSeqlensOptional != nullptr;
    const bool isInternalLayout = parsedLayout == Layout::BNSD || isVarLen;
    const auto qShape = q->GetViewShape();
    const int64_t batch = parsedLayout == Layout::TND ? 1 : qShape.GetDim(0);
    const int64_t seqlen = parsedLayout == Layout::TND ? qShape.GetDim(0) :
        qShape.GetDim(parsedLayout == Layout::BNSD ? 2 : 1);
    const int64_t headNum = parsedLayout == Layout::TND ? qShape.GetDim(1) :
        qShape.GetDim(parsedLayout == Layout::BNSD ? 1 : 2);
    const int64_t headDim =
        qShape.GetDim(parsedLayout == Layout::TND ? 2 : 3);

    const aclTensor *qBnsd = q;
    const aclTensor *kBnsd = k;
    const aclTensor *gkBnsd = gk;
    const aclTensor *betaBns = beta;
    const aclTensor *dAqkBnst = dAqk;
    const aclTensor *dAkkBnst = dAkk;
    const aclTensor *dqBnsd = dq;
    const aclTensor *dkBnsd = dk;
    const aclTensor *dbBns = db;
    const aclTensor *dgBnsd = dg;
    const aclTensor *dqOutBnsd = dqOut;
    const aclTensor *dkOutBnsd = dkOut;
    const aclTensor *dbOutBns = dbOut;
    const aclTensor *dgOutBnsd = dgOut;

    if (parsedLayout == Layout::TND) {
        const op::Shape vectorShape =
            MakeShape({1, seqlen, headNum, headDim});
        const op::Shape scalarShape =
            MakeShape({1, seqlen, headNum});
        const op::Shape matrixShape =
            MakeShape({1, seqlen, headNum, chunkSize});
        qBnsd = l0op::Reshape(q, vectorShape, executorPtr);
        kBnsd = l0op::Reshape(k, vectorShape, executorPtr);
        gkBnsd = l0op::Reshape(gk, vectorShape, executorPtr);
        betaBns = l0op::Reshape(beta, scalarShape, executorPtr);
        dAqkBnst = l0op::Reshape(dAqk, matrixShape, executorPtr);
        dAkkBnst = l0op::Reshape(dAkk, matrixShape, executorPtr);
        dqBnsd = l0op::Reshape(dq, vectorShape, executorPtr);
        dkBnsd = l0op::Reshape(dk, vectorShape, executorPtr);
        dbBns = l0op::Reshape(db, scalarShape, executorPtr);
        dgBnsd = l0op::Reshape(dg, vectorShape, executorPtr);
        dqOutBnsd = l0op::Reshape(dqOut, vectorShape, executorPtr);
        dkOutBnsd = l0op::Reshape(dkOut, vectorShape, executorPtr);
        dbOutBns = l0op::Reshape(dbOut, scalarShape, executorPtr);
        dgOutBnsd = l0op::Reshape(dgOut, vectorShape, executorPtr);
        CHECK_RET(qBnsd != nullptr && kBnsd != nullptr && gkBnsd != nullptr &&
                      betaBns != nullptr && dAqkBnst != nullptr &&
                      dAkkBnst != nullptr && dqBnsd != nullptr &&
                      dkBnsd != nullptr && dbBns != nullptr &&
                      dgBnsd != nullptr && dqOutBnsd != nullptr &&
                      dkOutBnsd != nullptr && dbOutBns != nullptr &&
                      dgOutBnsd != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    } else if (!isInternalLayout) {
        const op::Shape vectorShape = MakeShape({batch, headNum, seqlen, headDim});
        const op::Shape scalarShape = MakeShape({batch, headNum, seqlen});
        const op::Shape matrixShape = MakeShape({batch, headNum, seqlen, chunkSize});
        qBnsd = executorPtr->AllocTensor(vectorShape, q->GetDataType(), Format::FORMAT_ND);
        kBnsd = executorPtr->AllocTensor(vectorShape, k->GetDataType(), Format::FORMAT_ND);
        gkBnsd = executorPtr->AllocTensor(vectorShape, gk->GetDataType(), Format::FORMAT_ND);
        betaBns = executorPtr->AllocTensor(scalarShape, beta->GetDataType(), Format::FORMAT_ND);
        dAqkBnst = executorPtr->AllocTensor(matrixShape, dAqk->GetDataType(), Format::FORMAT_ND);
        dAkkBnst = executorPtr->AllocTensor(matrixShape, dAkk->GetDataType(), Format::FORMAT_ND);
        dqBnsd = executorPtr->AllocTensor(vectorShape, dq->GetDataType(), Format::FORMAT_ND);
        dkBnsd = executorPtr->AllocTensor(vectorShape, dk->GetDataType(), Format::FORMAT_ND);
        dbBns = executorPtr->AllocTensor(scalarShape, db->GetDataType(), Format::FORMAT_ND);
        dgBnsd = executorPtr->AllocTensor(vectorShape, dg->GetDataType(), Format::FORMAT_ND);
        dqOutBnsd = executorPtr->AllocTensor(vectorShape, dqOut->GetDataType(), Format::FORMAT_ND);
        dkOutBnsd = executorPtr->AllocTensor(vectorShape, dkOut->GetDataType(), Format::FORMAT_ND);
        dbOutBns = executorPtr->AllocTensor(scalarShape, dbOut->GetDataType(), Format::FORMAT_ND);
        dgOutBnsd = executorPtr->AllocTensor(vectorShape, dgOut->GetDataType(), Format::FORMAT_ND);
        CHECK_RET(qBnsd != nullptr && kBnsd != nullptr && gkBnsd != nullptr &&
                      betaBns != nullptr && dAqkBnst != nullptr && dAkkBnst != nullptr &&
                      dqBnsd != nullptr && dkBnsd != nullptr && dbBns != nullptr &&
                      dgBnsd != nullptr && dqOutBnsd != nullptr && dkOutBnsd != nullptr &&
                      dbOutBns != nullptr && dgOutBnsd != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);

        CHECK_RET(l0op::KdaLayoutSwap12(q, qBnsd, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(k, kBnsd, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(gk, gkBnsd, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(beta, betaBns, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(dAqk, dAqkBnst, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(dAkk, dAkkBnst, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(dq, dqBnsd, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(dk, dkBnsd, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(db, dbBns, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(dg, dgBnsd, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }

    auto result = l0op::ChunkKdaBwdIntra(
        qBnsd, kBnsd, gkBnsd, betaBns, dAqkBnst, dAkkBnst,
        dqBnsd, dkBnsd, dbBns, dgBnsd,
        cuSeqlensOptional, chunkIndicesOptional, chunkSize, safeGate,
        isVarLen ? 1 : 0, totalChunks,
        dqOutBnsd, dkOutBnsd, dbOutBns, dgOutBnsd, executorPtr);
    for (const aclTensor *tensor : result) {
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (!isInternalLayout) {
        CHECK_RET(l0op::KdaLayoutSwap12(result[0], nullptr, dqOut, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[1], dqOut, dkOut, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[2], dkOut, dbOut, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::KdaLayoutSwap12(result[3], dbOut, dgOut, executorPtr)[0] != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkKdaBwdIntra(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaBwdIntra);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkKdaBwdIntra launch failed.");
    return ACLNN_SUCCESS;
}
