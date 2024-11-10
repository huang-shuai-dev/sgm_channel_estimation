import torch

# 创建一个复数张量
real = torch.tensor([1, -2, 3, -4], dtype=torch.float32)
imag = torch.tensor([3, -4, 5, -6], dtype=torch.float32)
y = torch.complex(real, imag)

# 输出原始复数张量
print("Original complex tensor:", y)

# 条件判断和修改实部
real_mask = y.real > 0
y.real = torch.where(real_mask, torch.ones_like(y.real), -torch.ones_like(y.real))

# 条件判断和修改虚部
imag_mask = y.imag > 0
y.imag = torch.where(imag_mask, torch.ones_like(y.imag), -torch.ones_like(y.imag))

# 输出修改后的复数张量
print("Modified complex tensor:", y)