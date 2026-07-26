/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SITU_GLU_TORCH_ADPT_H
#define SITU_GLU_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor situ_glu(
    const at::Tensor& x,
    int64_t dim,
    double beta,
    double linear_beta,
    bool activate_left)
{
    TORCH_CHECK(x.dim() >= 1 && x.dim() <= 8,
                "situ_glu: x rank must be in [1, 8], but got ", x.dim());
    TORCH_CHECK(x.scalar_type() == at::kFloat ||
                    x.scalar_type() == at::kHalf ||
                    x.scalar_type() == at::kBFloat16,
                "situ_glu: x must be float32, float16, or bfloat16, but got ",
                x.scalar_type());

    const int64_t normalized_dim = dim < 0 ? dim + x.dim() : dim;
    TORCH_CHECK(normalized_dim >= 0 && normalized_dim < x.dim(),
                "situ_glu: dim must be in [", -x.dim(), ", ", x.dim() - 1,
                "], but got ", dim);
    TORCH_CHECK(x.size(normalized_dim) % 2 == 0,
                "situ_glu: x size at dim ", dim, " must be even, but got ",
                x.size(normalized_dim));

    std::vector<int64_t> y_shape(x.sizes().begin(), x.sizes().end());
    y_shape[normalized_dim] /= 2;
    at::Tensor y = at::empty(y_shape, x.options());

    EXEC_NPU_CMD(aclnnSituGlu,
                 x,
                 dim,
                 beta,
                 linear_beta,
                 activate_left,
                 y);
    return y;
}

}  // namespace vllm_ascend

#endif  // SITU_GLU_TORCH_ADPT_H
