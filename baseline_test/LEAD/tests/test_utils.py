"""
工具函数模块单元测试。
"""

import json
import os
import tempfile
import unittest

from lead.utils import (
    resolve_device,
    ensure_dir,
    save_jsonl,
    save_json,
    load_json,
    format_duration,
    Timer,
)


class TestResolveDevice(unittest.TestCase):
    """测试设备解析。"""

    def test_cpu_explicit(self):
        """显式指定 cpu。"""
        device = resolve_device("cpu")
        self.assertEqual(str(device), "cpu")

    def test_auto(self):
        """auto 应返回有效设备。"""
        device = resolve_device("auto")
        self.assertIn(str(device), ["cpu", "cuda"])


class TestEnsureDir(unittest.TestCase):
    """测试目录创建。"""

    def test_create_new_dir(self):
        """应成功创建新目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "sub", "dir")
            result = ensure_dir(new_dir)
            self.assertTrue(os.path.isdir(new_dir))
            self.assertEqual(result, new_dir)

    def test_existing_dir_no_error(self):
        """已存在的目录不应报错。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ensure_dir(tmpdir)


class TestJsonIO(unittest.TestCase):
    """测试 JSON/JSONL 读写。"""

    def test_save_load_json(self):
        """验证 JSON 写入和读取。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            data = {"accuracy": 0.85, "total": 100}
            save_json(data, path)
            loaded = load_json(path)
            self.assertEqual(loaded["accuracy"], 0.85)

    def test_save_jsonl(self):
        """验证 JSONL 写入。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.jsonl")
            data = [{"id": 0}, {"id": 1}, {"id": 2}]
            save_jsonl(data, path)
            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(json.loads(lines[0])["id"], 0)


class TestFormatDuration(unittest.TestCase):
    """测试时间格式化。"""

    def test_seconds(self):
        """小于 60 秒显示 xs。"""
        self.assertEqual(format_duration(5.2), "5.2s")

    def test_minutes(self):
        """60-3600 秒显示 xm xs。"""
        self.assertEqual(format_duration(125), "2m 5s")

    def test_hours(self):
        """大于 3600 秒显示 xh xm xs。"""
        self.assertEqual(format_duration(3725), "1h 2m 5s")


class TestTimer(unittest.TestCase):
    """测试计时器。"""

    def test_context_manager(self):
        """with 语句应正确计时。"""
        import time
        with Timer("test") as t:
            time.sleep(0.05)
        self.assertGreater(t.elapsed, 0.04)

    def test_manual_start_stop(self):
        """手动 start/stop 应正确计时。"""
        import time
        t = Timer("manual")
        t.start()
        time.sleep(0.05)
        elapsed = t.stop()
        self.assertGreater(elapsed, 0.04)


if __name__ == "__main__":
    unittest.main()
