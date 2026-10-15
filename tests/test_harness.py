"""基准设施测试：dummy 输入形状、统计、结果落盘的溯源字段。

用超小的自建 OpenVINO 模型做夹具，因此**不依赖** 224 MB 的 model_q4.onnx 或 NPU，
在没有大件的新 checkout 上也能跑。
"""
import json

import numpy as np
import openvino as ov
import openvino.opset13 as ops
import pytest

import harness as H


@pytest.fixture
def tiny_model():
    p = ops.parameter([-1, -1], np.int64, name="input_ids")
    q = ops.parameter([-1, -1], np.int64, name="attention_mask")
    lp = ops.parameter([-1, -1], np.int64, name="label_pos")
    body = ops.add(p, q)
    # label_pos 作为独立输出：真实模型里它的宽度（18）与序列长度（768）无关，
    # 夹具必须保持两者形状解耦，否则 reshape 测试反映不出真实约束
    # （最初的夹具把 label_pos 加进主链，[1,32]+[1,8] 会直接 reshape 失败）。
    return ov.Model([body, lp], [p, q, lp], "tiny")


def test_feeds_respect_names_and_dynamic_dims(tiny_model):
    feeds = H.feeds_for(tiny_model, seq_len=64, batch=2)
    assert set(feeds) == {"input_ids", "attention_mask", "label_pos"}
    for v in feeds.values():
        assert v.shape == (2, 64)
        assert v.dtype == np.int64
    assert feeds["attention_mask"].min() == 1, "mask 应为全 1"
    assert feeds["label_pos"].max() == 0, "label_pos 语义上是位置，基准里置 0"


def test_feeds_respect_static_dims(tiny_model):
    m = tiny_model
    m.reshape({"input_ids": [1, 32], "attention_mask": [1, 32], "label_pos": [1, 8]})
    feeds = H.feeds_for(m, seq_len=999)          # 静态维度应覆盖 seq_len 参数
    assert feeds["input_ids"].shape == (1, 32)
    assert feeds["label_pos"].shape == (1, 8), "静态维度必须沿用模型自身，不能被 seq_len 覆盖"


def test_feeds_are_reproducible_with_seed(tiny_model):
    a = H.feeds_for(tiny_model, seq_len=8, seed=1)["input_ids"]
    b = H.feeds_for(tiny_model, seq_len=8, seed=1)["input_ids"]
    np.testing.assert_array_equal(a, b)


def test_max_abs_err():
    ref = [np.zeros((2, 2), np.float32)]
    got = [np.full((2, 2), 0.5, np.float32)]
    assert H.max_abs_err(ref, got) == pytest.approx(0.5)


def test_order_devices_is_stable():
    assert H.order_devices(["GPU.1", "NPU", "CPU"]) == ["NPU", "CPU", "GPU.1"]
    assert H.order_devices(["GPU.1", "GPU.0"]) == ["GPU.0", "GPU.1"]


def test_run_on_cpu_timings(tiny_model):
    feeds = H.feeds_for(tiny_model, seq_len=16)
    st = H.run_on(H.make_core(), tiny_model, "CPU", feeds, repeat=3, warmup=1)
    assert "error" not in st, st
    assert st["runs"] == 3
    assert st["min_ms"] <= st["p50_ms"] <= st["max_ms"]
    assert st["compile_s"] is not None


def test_run_on_bad_device_captures_error(tiny_model):
    """设备失败必须变成 {"error": ...}，而不是让整个基准崩掉。"""
    feeds = H.feeds_for(tiny_model, seq_len=16)
    st = H.run_on(H.make_core(), tiny_model, "NO_SUCH_DEVICE", feeds, repeat=1)
    assert "error" in st


def test_write_result_provenance(tmp_path):
    path = H.write_result("x.json", {"a": 1}, outdir=tmp_path, script="tests/test_harness.py")
    d = json.loads(open(path, encoding="utf-8").read())
    assert d["a"] == 1
    assert d["generated_by"] == "tests/test_harness.py"
    assert "generated_at" in d and "openvino" in d and "host" in d


def test_write_result_encodes_numpy(tmp_path):
    path = H.write_result("y.json", {"arr": np.arange(3), "n": np.int64(7)}, outdir=tmp_path)
    d = json.loads(open(path, encoding="utf-8").read())
    assert d["arr"] == [0, 1, 2] and d["n"] == 7
