/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CHUNK_KDA_BWD_INTRA_ARCH35_VECTOR_H
#define CHUNK_KDA_BWD_INTRA_ARCH35_VECTOR_H

// The orchestration and GM/UB pipeline are intentionally shared with A2/A3.
// All A5 Vector arithmetic and cross-core synchronization are selected inside
// the common class at compile time and implemented by the RegBase helpers.
#include "../chunk_kda_bwd_intra_vector.h"

#endif // CHUNK_KDA_BWD_INTRA_ARCH35_VECTOR_H
