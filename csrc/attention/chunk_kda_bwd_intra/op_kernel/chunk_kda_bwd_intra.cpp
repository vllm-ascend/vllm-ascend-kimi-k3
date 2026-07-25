/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

#include "chunk_kda_bwd_intra_struct.h"
#include "chunk_kda_bwd_intra_tiling_key.h"
#include "chunk_kda_bwd_intra_common.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_kda_bwd_intra_cube.h"
#include "arch35/chunk_kda_bwd_intra_vector.h"
#else
#include "chunk_kda_bwd_intra_cube.h"
#include "chunk_kda_bwd_intra_vector.h"
#endif

namespace KDA {

template <uint32_t K_KEY>
struct HeadDim;
template <>
struct HeadDim<CHUNK_KDA_BWD_INTRA_K64> {
    static constexpr uint32_t value = 64;
};
template <>
struct HeadDim<CHUNK_KDA_BWD_INTRA_K128> {
    static constexpr uint32_t value = 128;
};
template <>
struct HeadDim<CHUNK_KDA_BWD_INTRA_K256> {
    static constexpr uint32_t value = 256;
};

template <uint32_t K_DIM, uint32_t CHUNK_SIZE, bool SAFE_GATE, bool VARLEN_TND>
__aicore__ inline void ChunkKdaBwdIntraKernelImpl(
    GM_ADDR q, GM_ADDR k, GM_ADDR gk, GM_ADDR beta, GM_ADDR dAqk, GM_ADDR dAkk,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR db, GM_ADDR dg, GM_ADDR dqOut, GM_ADDR dkOut,
    GM_ADDR dbOut, GM_ADDR dgOut, GM_ADDR chunkMetadata, GM_ADDR workspace,
    const ChunkKdaBwdIntraTilingData *tiling)
{
    if ASCEND_IS_AIC {
        ChunkKdaBwdIntraCubeProcess<K_DIM, CHUNK_SIZE, SAFE_GATE, VARLEN_TND> cube(
            chunkMetadata, workspace);
        cube.Init(*tiling);
        cube.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        ChunkKdaBwdIntraVectorProcess<K_DIM, CHUNK_SIZE, SAFE_GATE, VARLEN_TND> vector(
            q, k, gk, beta, dAqk, dAkk, dq, dk, db, dg, dqOut, dkOut,
            dbOut, dgOut, chunkMetadata, workspace);
        vector.Init(*tiling, &pipe);
        vector.Process();
    }
}

} // namespace KDA

#ifndef TORCH_MODE
template <uint32_t K_KEY, uint32_t SAFE_KEY, uint32_t LAYOUT_KEY>
__global__ __aicore__ void chunk_kda_bwd_intra(
    GM_ADDR q, GM_ADDR k, GM_ADDR gk, GM_ADDR beta, GM_ADDR dAqk, GM_ADDR dAkk,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR db, GM_ADDR dg, GM_ADDR cu_seqlens,
    GM_ADDR chunk_metadata, GM_ADDR dq_out, GM_ADDR dk_out, GM_ADDR db_out,
    GM_ADDR dg_out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)cu_seqlens;
    AscendC::AscendCUtils::SetOverflow(1);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(KDA::ChunkKdaBwdIntraTilingData);
    GET_TILING_DATA_WITH_STRUCT(KDA::ChunkKdaBwdIntraTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    constexpr uint32_t kDim = KDA::HeadDim<K_KEY>::value;
    constexpr bool safeGate = SAFE_KEY == CHUNK_KDA_BWD_INTRA_SAFE;
    constexpr bool varLenTnd = LAYOUT_KEY == CHUNK_KDA_BWD_INTRA_VARLEN_TND;
    KDA::ChunkKdaBwdIntraKernelImpl<kDim, KDA::kChunkSize, safeGate, varLenTnd>(
        q, k, gk, beta, dAqk, dAkk, dq, dk, db, dg, dq_out, dk_out,
        db_out, dg_out, chunk_metadata, userWorkspace, &tilingData);
}
#endif
