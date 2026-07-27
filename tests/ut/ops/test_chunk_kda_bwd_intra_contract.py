# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Static regressions for the migrated chunk KDA intra-backward operator."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OP_ROOT = ROOT / "csrc/attention/chunk_kda_bwd_intra"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_operator_tree_contains_host_kernel_and_arch35_paths():
    required = (
        "CMakeLists.txt",
        "op_host/CMakeLists.txt",
        "op_host/chunk_kda_bwd_intra_def.cpp",
        "op_host/chunk_kda_bwd_intra_tiling_processor.h",
        "op_host/op_api/aclnn_chunk_kda_bwd_intra.cpp",
        "op_host/op_api/aclnn_chunk_kda_bwd_intra.h",
        "op_host/op_tiling/chunk_kda_bwd_intra_tiling.cpp",
        "op_kernel/chunk_kda_bwd_intra.cpp",
        "op_kernel/chunk_kda_bwd_intra_cube.h",
        "op_kernel/chunk_kda_bwd_intra_vector.h",
        "op_kernel/arch35/chunk_kda_bwd_intra_cube.h",
        "op_kernel/arch35/chunk_kda_bwd_intra_regbase.h",
        "op_kernel/arch35/chunk_kda_bwd_intra_vector.h",
    )

    assert all((OP_ROOT / path).is_file() for path in required)


def test_migrated_sources_use_huawei_cann_notice():
    sources = [path for path in OP_ROOT.rglob("*") if path.is_file()]
    sources.append(ROOT / "csrc/moe/common/kernel_utils/vector/regbase.hpp")

    for path in sources:
        assert "Copyright (c) 2026 Huawei Technologies Co., Ltd." in _read(path)[:1200], path
        assert "CANN Open Software License Agreement Version 2.0" in _read(path)[:1200], path


def test_build_wires_catlass_common_headers_and_supported_socs():
    cmake = _read(OP_ROOT / "op_host/CMakeLists.txt")
    build_script = _read(ROOT / "csrc/build_aclnn.sh")
    op_def = _read(OP_ROOT / "op_host/chunk_kda_bwd_intra_def.cpp")

    assert "CATLASS_INCLUDE_DIR_ABS" in cmake
    assert "COMMON_KERNEL_UTILS_DIR_ABS" in cmake
    assert "if (NOT BUILD_OPS_RTY_KERNEL)" in cmake
    assert build_script.count('"chunk_kda_bwd_intra"') == 3
    for soc in ("ascend910b", "ascend910_93", "ascend950"):
        assert f'AddConfig("{soc}", config)' in op_def


def test_arch35_regbase_dependency_is_packaged():
    arch35 = _read(OP_ROOT / "op_kernel/arch35/chunk_kda_bwd_intra_regbase.h")
    common_header = ROOT / "csrc/moe/common/kernel_utils/vector/regbase.hpp"

    assert '#include "kernel_utils/vector/regbase.hpp"' in arch35
    assert common_header.is_file()
    assert "using namespace AscendC::MicroAPI;" in _read(common_header)


def test_torch_schema_and_meta_are_registered_once():
    binding = _read(ROOT / "csrc/torch_binding.cpp")
    meta = _read(ROOT / "csrc/torch_binding_meta.cpp")

    schema = (
        "chunk_kda_bwd_intra(Tensor q, Tensor k, Tensor gk, Tensor beta, "
        "Tensor dAqk, Tensor dAkk, Tensor dq, Tensor dk, Tensor db, Tensor dg"
    )
    assert binding.count(schema) == 1
    assert "int[]? cu_seqlens=None" in binding
    assert "int[]? chunk_indices=None" in binding
    assert "int chunk_size=64" in binding
    assert "bool safe_gate=True" in binding
    assert binding.count('ops.impl("chunk_kda_bwd_intra", torch::kPrivateUse1') == 1
    assert meta.count('ops.impl("chunk_kda_bwd_intra", &vllm_ascend::meta::chunk_kda_bwd_intra_meta)') == 1


def test_aclnn_contract_preserves_four_gradient_outputs():
    header = _read(OP_ROOT / "op_host/op_api/aclnn_chunk_kda_bwd_intra.h")
    source = _read(OP_ROOT / "op_host/op_api/aclnn_chunk_kda_bwd_intra.cpp")

    for output in ("dqOut", "dkOut", "dbOut", "dgOut"):
        assert f"const aclTensor *{output}" in header
    assert "safe_gate=false is reserved but not supported in v1." in source
    assert "varlen supports K=128; dense supports K=64/128/256." in source
    assert "chunk_indices must use canonical sequence-major order." in source


def test_internal_layout_aclnn_paths_build_rank4_views_before_custom_op_launch():
    source = _read(OP_ROOT / "op_host/op_api/aclnn_chunk_kda_bwd_intra.cpp")

    assert '#include "aclnn_kernels/reshape.h"' in source
    assert (
        "if (parsedLayout == Layout::TND || parsedLayout == Layout::BNSD)"
        in source
    )
    assert "MakeShape({batch, headNum, seqlen, headDim})" in source
    assert "MakeShape({batch, headNum, seqlen})" in source
    assert "MakeShape({batch, headNum, seqlen, chunkSize})" in source
    for tensor in (
        "q",
        "k",
        "gk",
        "beta",
        "dAqk",
        "dAkk",
        "dq",
        "dk",
        "db",
        "dg",
        "dqOut",
        "dkOut",
        "dbOut",
        "dgOut",
    ):
        assert f"l0op::Reshape({tensor}," in source
