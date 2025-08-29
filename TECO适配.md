# einops

## 1. 功能概述
einops 提供了一种简洁、直观且无错误的方式来进行张量操作（如 reshape、transpose、reduce 等）。通过统一的语法 rearrange(pattern)，它让维度变换变得可读性强、易于理解，无需手动计算维度，避免常见的形状错误，广泛适用于 PyTorch、NumPy、TensorFlow 等框架。

- 参考实现：
    ```
    url=https://github.com/arogozhnikov/einops
    commit_id=361b11e87da94ead4bd09de636c5dbed73e0e3e0
    ```

## 2. 快速开始
下面介绍此三方库的安装方法、安装验证以及使用方法：
### 2.1 三方库安装
    pip install einops

### 2.2 安装验证

- 验证einops的功能请运行测试脚本
    ```
    python -m einops.tests.run_tests torch numpy xxx --pip-install
    ```
其中xxx可以从['numpy', 'torch', 'jax', 'tensorflow', 'cupy', 'paddle', 'oneflow', 'pytensor']选择需要验证的框架。--pip-install非必填项，添加后会自动安装依赖项。

### 2.3 使用方法
- 首先导入einops库：
    ```
    from einops import rearrange, reduce, repeat
    ```
- 下面以输入x维度为(2, 3, 32, 32)为例介绍einops的使用方法：
#### 2.3.1 重塑
- 改变张量结构，等价于 view/reshape，输出维度(2, 96, 32)
    ```
    rearrange(x, 'b c h w -> b (c h) w')
    ```
#### 2.3.2 转置
- 交换维度顺序，输出维度(2, 32, 32, 3)
    ```
    rearrange(x, 'b c h w -> b h w c')
    ```
#### 2.3.3 拆分维度
- 将一个大维度拆成多个，输出维度(2, 3, 4, 4, 8, 8)
    ```
    rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=8, p2=8)
    ```
#### 2.3.4 合并维度	
- 	将多个维度合并成一个，输出维度(2, 3072)
    ```
    rearrange(x, 'b c h w -> b (c h w)')
    ```
#### 2.3.5 空间展平
- 将 H×W 空间展平为序列（如 ViT），输出维度(2, 1024, 3)
    ```
    rearrange(x, 'b c h w -> b (h w) c')
    ```
#### 2.3.6 图像切块
- 常用于 Vision Transformer，输出维度(2, 4, 768)
    ```
    rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=16, p2=16)
    ```
#### 2.3.7 添加新维度
- 相当于 unsqueeze，输出维度(2, 3, 32, 32, 1)
    ```
    rearrange(x, 'b c h w -> b c h w 1')
    ```
## 3. teco适配情况
- 将输入张量移动到GPU后，使用unsqueeze占用显存，因此unsqueeze能在teco加速卡正常运行
- 使用2.2节介绍的测试脚本显示报错：FAILED test_other.py::test_torch_compile_for_layers - torch._dynamo.exc.InternalTorchDynamoError: 'torch_sdaa._C._SDAADeviceProperties' object has no attribute 'major'，即einops.rearrange不能被torch.compile编译，但rearrange功能正常可正常使用。
