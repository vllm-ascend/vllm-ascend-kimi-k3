# SituGlu

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |    √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |    ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：SiTU 门控线性单元（SiTU Gated Linear Unit）激活函数。对输入张量 x 沿指定维度切分为门控（gate）与上路径（up）两半，按 SiTU 公式计算输出。

- 计算流程：

  对给定的输入张量 x，其维度为 [a, b, c, d, e, f, g, ...]，算子 SituGlu 对其进行以下计算：

  1. 将 x 基于输入参数 dim 进行合轴，合轴后维度为 [pre, cut]，其中 cut 必须为偶数。令 h = cut // 2。

  2. 根据输入参数 activate_left 对 x 进行前后切分：

     - activate_left 为 true 时（默认）：

     $$
     gate = x[:, :h], \quad up = x[:, h:]
     $$

     - activate_left 为 false 时：

     $$
     gate = x[:, h:], \quad up = x[:, :h]
     $$

  3. 根据输入参数 beta、linear_beta 进行 SiTU 计算：

     $$
     situ\_a = \beta \cdot \tanh\left(\frac{gate}{\beta}\right) \cdot \text{sigmoid}(gate)
     $$

     当 linear_beta > 0 时：

     $$
     up = linear\_beta \cdot \tanh\left(\frac{up}{linear\_beta}\right)
     $$

  4. 输出：

     $$
     y = situ\_a \cdot up
     $$

  5. 重塑输出张量 y 的维度数量与合轴前的 x 的维度数量一致，dim 轴上的大小为 x 的一半，其他维度与 x 相同。

## 约束说明

- 输入数据类型支持 FLOAT（float32）、FLOAT16（fp16）、BFLOAT16（bf16），输出数据类型与输入保持一致；fp16/bf16 输入时算子内部统一转为 float32 计算后再转回原数据类型。
- 输入 x 在 dim 维度上的大小必须为偶数。
- dim 取值范围为 [-x.dim(), x.dim()-1]，默认 -1。

## 接口

- aclnn 接口：[aclnnSituGlu](./docs/aclnnSituGlu.md)
