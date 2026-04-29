"""
提示模板模块单元测试。
"""

import unittest

from lead.prompts import (
    format_mcq_prompt,
    format_open_ended_prompt,
    format_prompt_from_sample,
    build_chat_messages,
)


class TestFormatMCQPrompt(unittest.TestCase):
    """测试选择题提示格式化。"""

    def test_basic_format(self):
        """验证基本 MCQ 格式。"""
        result = format_mcq_prompt("What is 1+1?", "A. 1\nB. 2\nC. 3\nD. 4")
        self.assertIn("What is 1+1?", result)
        self.assertIn("Options:", result)
        self.assertIn("A. 1", result)

    def test_cot_format(self):
        """CoT 模式应包含引导语。"""
        result = format_mcq_prompt("Q?", "A. x\nB. y", use_cot=True)
        self.assertIn("step by step", result)

    def test_non_cot_no_guide(self):
        """非 CoT 模式不应包含引导语。"""
        result = format_mcq_prompt("Q?", "A. x\nB. y", use_cot=False)
        self.assertNotIn("step by step", result)


class TestFormatOpenEndedPrompt(unittest.TestCase):
    """测试开放式问答提示格式化。"""

    def test_basic_format(self):
        """验证基本开放式格式。"""
        result = format_open_ended_prompt("Explain Newton's second law.")
        self.assertIn("Explain Newton's second law.", result)

    def test_cot_format(self):
        """CoT 模式应包含引导语。"""
        result = format_open_ended_prompt("Explain.", use_cot=True)
        self.assertIn("step by step", result)


class TestFormatPromptFromSample(unittest.TestCase):
    """测试从样本字典自动格式化。"""

    def test_mcq_sample(self):
        """包含 options 的样本应使用 MCQ 模板。"""
        sample = {"question": "Q?", "options": "A. x\nB. y"}
        result = format_prompt_from_sample(sample)
        self.assertIn("Options:", result)

    def test_open_ended_sample(self):
        """不含 options 的样本应使用 Open-Ended 模板。"""
        sample = {"question": "Derive the formula.", "options": ""}
        result = format_prompt_from_sample(sample)
        self.assertIn("Derive the formula.", result)
        self.assertNotIn("Options:", result)


class TestBuildChatMessages(unittest.TestCase):
    """测试对话消息构建。"""

    def test_basic_messages(self):
        """验证消息结构。"""
        msgs = build_chat_messages("img.jpg", "Describe this.")
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")
        content = msgs[0]["content"]
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0]["type"], "image")
        self.assertEqual(content[1]["type"], "text")

    def test_with_system_prompt(self):
        """带 system prompt 时应有两条消息。"""
        msgs = build_chat_messages("img.jpg", "Q?", system_prompt="You are helpful.")
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")


if __name__ == "__main__":
    unittest.main()
