"""
数据加载模块单元测试。
"""

import json
import os
import tempfile
import unittest

from lead.data import (
    load_jsonl,
    resolve_image_paths,
    filter_by_subtopic,
    filter_by_difficulty,
    filter_by_language,
    load_dataset,
    get_dataset_statistics,
)


class TestLoadJsonl(unittest.TestCase):
    """测试 JSONL 加载功能。"""

    def setUp(self):
        """创建临时 JSONL 文件。"""
        self.tmpdir = tempfile.mkdtemp()
        self.jsonl_path = os.path.join(self.tmpdir, "test.jsonl")
        samples = [
            {"id": 0, "image": "images/0.jpg", "subtopic": "Mechanics",
             "difficulty": 2, "language": "english"},
            {"id": 1, "image": "images/1.jpg", "subtopic": "Optics",
             "difficulty": 4, "language": "chinese"},
            {"id": 2, "image": "images/2.jpg", "subtopic": "Mechanics",
             "difficulty": 3, "language": "english"},
        ]
        with open(self.jsonl_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

    def test_load_jsonl_basic(self):
        """验证基本加载功能和样本数量。"""
        data = load_jsonl(self.jsonl_path)
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0]["id"], 0)

    def test_load_jsonl_not_found(self):
        """不存在的文件应抛出 FileNotFoundError。"""
        with self.assertRaises(FileNotFoundError):
            load_jsonl("/nonexistent/path.jsonl")

    def test_resolve_image_paths(self):
        """验证相对路径解析为绝对路径。"""
        data = load_jsonl(self.jsonl_path)
        resolve_image_paths(data, self.tmpdir)
        expected = os.path.join(self.tmpdir, "images/0.jpg")
        self.assertEqual(data[0]["image"], expected)

    def test_resolve_absolute_path_unchanged(self):
        """绝对路径不应被修改。"""
        data = [{"image": "/absolute/path/img.jpg"}]
        resolve_image_paths(data, "/some/dir")
        self.assertEqual(data[0]["image"], "/absolute/path/img.jpg")


class TestFilters(unittest.TestCase):
    """测试数据过滤功能。"""

    def setUp(self):
        """创建测试样本集。"""
        self.samples = [
            {"id": 0, "subtopic": "Mechanics", "difficulty": 2, "language": "english"},
            {"id": 1, "subtopic": "Optics", "difficulty": 4, "language": "chinese"},
            {"id": 2, "subtopic": "Mechanics", "difficulty": 3, "language": "english"},
            {"id": 3, "subtopic": "Thermodynamics", "difficulty": 5, "language": "chinese"},
        ]

    def test_filter_by_subtopic(self):
        """验证子领域过滤。"""
        result = filter_by_subtopic(self.samples, ["Mechanics"])
        self.assertEqual(len(result), 2)
        self.assertTrue(all(s["subtopic"] == "Mechanics" for s in result))

    def test_filter_by_difficulty(self):
        """验证难度过滤。"""
        result = filter_by_difficulty(self.samples, min_difficulty=3, max_difficulty=4)
        self.assertEqual(len(result), 2)
        self.assertIn(result[0]["difficulty"], [3, 4])

    def test_filter_by_language(self):
        """验证语言过滤。"""
        result = filter_by_language(self.samples, "chinese")
        self.assertEqual(len(result), 2)
        self.assertTrue(all(s["language"] == "chinese" for s in result))


class TestDatasetStatistics(unittest.TestCase):
    """测试数据集统计功能。"""

    def test_statistics(self):
        """验证统计结果的结构和数值。"""
        samples = [
            {"subtopic": "Mechanics", "difficulty": 2, "language": "english"},
            {"subtopic": "Mechanics", "difficulty": 3, "language": "english"},
            {"subtopic": "Optics", "difficulty": 4, "language": "chinese"},
        ]
        stats = get_dataset_statistics(samples)
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["subtopics"]["Mechanics"], 2)
        self.assertEqual(stats["subtopics"]["Optics"], 1)
        self.assertEqual(stats["languages"]["english"], 2)


if __name__ == "__main__":
    unittest.main()
