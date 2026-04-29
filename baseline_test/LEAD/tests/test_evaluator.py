"""
答案评估模块单元测试。
"""

import unittest

from lead.evaluator import (
    extract_mcq_answer,
    evaluate_single,
    evaluate_dataset,
)


class TestExtractMCQAnswer(unittest.TestCase):
    """测试选择题答案提取。"""

    def test_answer_is_pattern(self):
        """匹配 'The answer is X' 模式。"""
        self.assertEqual(extract_mcq_answer("The answer is A"), "A")
        self.assertEqual(extract_mcq_answer("The answer is (B)"), "B")
        self.assertEqual(extract_mcq_answer("the correct answer is C"), "C")

    def test_boxed_pattern(self):
        """匹配 LaTeX \\boxed{X} 模式。"""
        self.assertEqual(extract_mcq_answer("\\boxed{D}"), "D")

    def test_bold_pattern(self):
        """匹配 Markdown **X** 模式。"""
        self.assertEqual(extract_mcq_answer("So the answer is **B**"), "B")

    def test_answer_colon_pattern(self):
        """匹配 'Answer: X' 模式。"""
        self.assertEqual(extract_mcq_answer("Answer: C"), "C")
        self.assertEqual(extract_mcq_answer("answer: (A)"), "A")

    def test_trailing_letter(self):
        """从文本末尾提取独立字母。"""
        text = "After analysis, the best choice is D"
        self.assertEqual(extract_mcq_answer(text), "D")

    def test_empty_input(self):
        """空输入应返回 None。"""
        self.assertIsNone(extract_mcq_answer(""))
        self.assertIsNone(extract_mcq_answer(None))

    def test_no_match(self):
        """无法匹配时返回 None。"""
        self.assertIsNone(extract_mcq_answer("This is a random sentence."))


class TestEvaluateSingle(unittest.TestCase):
    """测试单样本评估。"""

    def test_correct_answer(self):
        """正确答案应返回 True。"""
        is_correct, extracted = evaluate_single("The answer is A", "A")
        self.assertTrue(is_correct)
        self.assertEqual(extracted, "A")

    def test_wrong_answer(self):
        """错误答案应返回 False。"""
        is_correct, extracted = evaluate_single("The answer is B", "A")
        self.assertFalse(is_correct)
        self.assertEqual(extracted, "B")

    def test_case_insensitive(self):
        """答案比较应忽略大小写。"""
        is_correct, _ = evaluate_single("The answer is a", "A")
        self.assertTrue(is_correct)


class TestEvaluateDataset(unittest.TestCase):
    """测试数据集级别的评估。"""

    def test_basic_accuracy(self):
        """验证准确率计算。"""
        dataset = [
            {"model_answer": "The answer is A", "answer": "A",
             "subtopic": "Mechanics", "difficulty": 2, "language": "english"},
            {"model_answer": "The answer is B", "answer": "A",
             "subtopic": "Optics", "difficulty": 3, "language": "english"},
            {"model_answer": "The answer is C", "answer": "C",
             "subtopic": "Mechanics", "difficulty": 2, "language": "chinese"},
        ]
        results = evaluate_dataset(dataset)
        self.assertAlmostEqual(results["accuracy"], 2 / 3)
        self.assertEqual(results["correct"], 2)
        self.assertEqual(results["total"], 3)

    def test_subtopic_breakdown(self):
        """验证按子领域的准确率拆分。"""
        dataset = [
            {"model_answer": "A", "answer": "A",
             "subtopic": "Mechanics", "difficulty": 1, "language": "english"},
            {"model_answer": "B", "answer": "A",
             "subtopic": "Mechanics", "difficulty": 1, "language": "english"},
        ]
        results = evaluate_dataset(dataset)
        mech = results["by_subtopic"]["Mechanics"]
        self.assertEqual(mech["total"], 2)
        self.assertEqual(mech["correct"], 1)

    def test_none_prediction(self):
        """None 预测应记为 failed_extraction。"""
        dataset = [
            {"model_answer": None, "answer": "A",
             "subtopic": "Optics", "difficulty": 1, "language": "english"},
        ]
        results = evaluate_dataset(dataset)
        self.assertEqual(results["failed_extraction"], 1)
        self.assertEqual(results["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
