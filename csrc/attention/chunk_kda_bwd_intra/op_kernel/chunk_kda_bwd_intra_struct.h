/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CHUNK_KDA_BWD_INTRA_STRUCT_H
#define CHUNK_KDA_BWD_INTRA_STRUCT_H

#include <cstdint>

namespace KDA {

struct ChunkKdaBwdIntraTilingData {
    int64_t batch;
    int64_t headNum;
    int64_t seqlen;
    int64_t headDim;
    int64_t chunkSize;
    int64_t chunkNum;
    int64_t chunkNumPerBatch;

    int64_t workspaceSlotSize;
    int64_t workspaceCoreSize;
    int64_t resultRegionOffset;

    int64_t aLowerOffset;
    int64_t bLowerOffset;
    int64_t aUpperOffset;
    int64_t bUpperOffset;
    int64_t resultDqOffset;
    int64_t resultDkLowerOffset;
    int64_t resultDkUpperOffset;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_INTRA_STRUCT_H
