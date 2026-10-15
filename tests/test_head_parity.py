"""验证 bench_jev 用 numpy 重写的 head 与 engine/model.py 的 torch 实现等价。

为什么必须测这一条：混合方案（NPU 只跑编码器、head 回 CPU）的整个数值可信度，
都建立在"numpy head == torch head"这个前提上。若两者不等价，
`jev-hybrid-final.json` 里的 `max_abs_err`（NPU 0.024 级）就无从解释——
那到底是 NPU 的数值误差，还是我自己重写 head 引入的偏差，就分不清了。

对照实现取自 engine/model.py:54-59：
    picked = h.gather(1, label_pos[..., None].expand(-1, -1, d))
    logits = Linear(d, 1)(dropout(picked)).squeeze(-1)
（推理时 dropout 恒等，故只比 Linear。）
"""
import numpy as np
import pytest
import torch

import paths
from bench_jev import head_logits, load_head

D, M, L = 768, 18, 32

# 容差说明：numpy 与 torch 走的是不同 BLAS 内核，float32 求和顺序不同，
# 绝对值 ~50 的 logit 上会出现 ~2e-5 的舍入差（相对误差 ~1e-6）。
# 这个量级比本项目真正要度量的 NPU 误差（0.024）小三个数量级，
# 因此下面所有 assert 都额外要求"远小于 NPU 误差"，确保 head 重写不会污染结论。
ABS_TOL = 1e-4


def _reference(h, pos, W, b):
    """engine/model.py 的 torch 版 head。"""
    picked = h.gather(1, pos.unsqueeze(-1).expand(-1, -1, h.size(-1)))
    return (picked @ W.T + b).squeeze(-1).numpy()


def test_numpy_head_matches_torch_bitwise_close():
    torch.manual_seed(0)
    h = torch.randn(1, L, D)
    pos = torch.randint(0, L, (1, M))
    W = torch.randn(1, D)
    b = torch.randn(1)
    ref = _reference(h, pos, W, b)          # [1, M]
    got = head_logits(h.numpy(), pos.numpy(), W.numpy(), b.numpy())   # [1, M]
    assert got.shape == ref.shape == (1, M)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=ABS_TOL)
    assert np.max(np.abs(got - ref)) < 0.024 / 20, "head 重写误差必须远小于 NPU 误差"


def test_head_actually_gathers_by_position():
    """交换 label_pos 必须改变输出，否则说明实现忽略了位置、退化成取前 M 个 token。"""
    torch.manual_seed(1)
    h = torch.randn(1, L, D)
    W = torch.randn(1, D)
    b = torch.randn(1)
    pos_a = torch.zeros(1, M, dtype=torch.long)
    pos_b = torch.arange(M, dtype=torch.long).unsqueeze(0)
    a = head_logits(h.numpy(), pos_a.numpy(), W.numpy(), b.numpy())
    bb = head_logits(h.numpy(), pos_b.numpy(), W.numpy(), b.numpy())
    assert not np.allclose(a, bb)


def test_numpy_head_matches_torch_for_batch_3():
    """批大小不受限：take_along_axis 与 torch.gather 在 B>1 时同样逐位对应。"""
    torch.manual_seed(2)
    h = torch.randn(3, L, D)
    pos = torch.randint(0, L, (3, M))
    W = torch.randn(1, D)
    b = torch.randn(1)
    ref = _reference(h, pos, W, b)
    got = head_logits(h.numpy(), pos.numpy(), W.numpy(), b.numpy())
    assert got.shape == (3, M)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=ABS_TOL)
    assert np.max(np.abs(got - ref)) < 0.024 / 20, "head 重写误差必须远小于 NPU 误差"


@pytest.mark.skipif(not paths.JEV_CKPT.exists(), reason="需要 output/onnx/jevlite.pt")
def test_real_checkpoint_head_tensors():
    W, b, _val_acc, max_len = load_head()
    assert W.shape == (1, 768) and b.shape == (1,)
    assert max_len == 768, "max_len 是 ONNX 静态形状的取值依据"
