# SPDX-License-Identifier: Apache-2.0
"""Source-level contract checks for the A5 SituGlu integration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OP_ROOT = ROOT / "csrc/moe/situ_glu"


def _read(path: Path) -> str:
    return path.read_text()


def test_situ_glu_is_built_only_for_a5():
    host_cmake = _read(OP_ROOT / "op_host/CMakeLists.txt")
    op_def = _read(OP_ROOT / "op_host/situ_glu_def.cpp")

    assert 'if(NOT "ascend950" IN_LIST ASCEND_COMPUTE_UNIT)' in host_cmake
    assert "add_modules_sources_with_soc(OPTYPE situ_glu ACLNNTYPE aclnn)" in host_cmake
    assert 'AddConfig("ascend950")' in op_def
    assert 'AddConfig("ascend910b")' not in op_def
    assert 'AddConfig("ascend910_93")' not in op_def


def test_situ_glu_reuses_repository_tiling_dependencies():
    tiling_header = _read(OP_ROOT / "op_host/situ_glu_tiling.h")
    infer_shape = _read(OP_ROOT / "op_host/situ_glu_infershape.cpp")

    assert '"op_host/tiling_base.h"' not in tiling_header
    assert '"op_host/tiling_templates_registry.h"' not in tiling_header
    assert '"op_host/tiling_util.h"' not in tiling_header
    for dependency in ("tiling_base.h", "tiling_templates_registry.h", "tiling_util.h"):
        assert f'"../../dequant_swiglu_quant/tiling_base/{dependency}"' in tiling_header
    assert '"error_util.h"' not in infer_shape
    assert "OPS_CHECK_NULL_WITH_CONTEXT" not in infer_shape
    assert "OP_CHECK_NULL_WITH_CONTEXT" in infer_shape


def test_situ_glu_torch_and_meta_contracts_are_registered():
    adapter = _read(OP_ROOT / "situ_glu_torch_adpt.h")
    binding = _read(ROOT / "csrc/torch_binding.cpp")
    meta = _read(ROOT / "csrc/torch_binding_meta.cpp")

    assert "EXEC_NPU_CMD(aclnnSituGlu" in adapter
    assert '"situ_glu(Tensor x, "' in binding
    assert '&vllm_ascend::situ_glu);' in binding
    assert "at::Tensor situ_glu_meta(" in meta
    assert 'ops.impl("situ_glu", &vllm_ascend::meta::situ_glu_meta);' in meta


def test_a5_w4a16_situ_uses_situ_glu_without_changing_w4a8_mx():
    moe_mlp = _read(ROOT / "vllm_ascend/ops/fused_moe/moe_mlp.py")
    w4a16_body = moe_mlp.split("def _w4a16_mxfp4_situ_apply_mlp(", 1)[1].split(
        "def _w4a8_situ_apply_mlp(", 1
    )[0]
    w4a8_body = moe_mlp.split("def _w4a8_situ_apply_mlp(", 1)[1].split(
        "def quant_apply_mlp(", 1
    )[0]

    assert "torch.ops._C_ascend.situ_glu(" in w4a16_body
    assert "situ_and_mul(" not in w4a16_body
    assert "torch.ops._C_ascend.situ_mx_quant(" in w4a8_body
    assert "torch.ops._C_ascend.situ_glu(" not in w4a8_body


def test_upstream_kernel_keeps_all_three_dtype_instantiations():
    entry = _read(OP_ROOT / "op_kernel/situ_glu.cpp")
    kernel = _read(OP_ROOT / "op_kernel/situ_glu.hpp")

    assert "SituGluBase<float>" in entry
    assert "SituGluBase<half>" in entry
    assert "SituGluBase<bfloat16_t>" in entry
    assert "SituIsBf16<bfloat16_t>" in kernel
    assert "linearBeta_ > 0.0f" in kernel
