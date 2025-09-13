import unittest
import torch
from tokenizer import SupernovaTokenizer, format_training_example

class TestLabelMasking(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SupernovaTokenizer()
        # Example messages with user and assistant turns
        self.messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi, how can I help you?"},
            {"role": "user", "content": "Tell me a joke."},
            {"role": "assistant", "content": "Why did the chicken cross the road?"}
        ]

    def test_at_least_one_unmasked_label(self):
        example = format_training_example(self.tokenizer, self.messages)
        labels = example["labels"]
        num_unmasked = (labels != -100).sum().item()
        self.assertGreater(num_unmasked, 0, "All labels are masked!")

    def test_warning_on_all_masked(self):
        # All user messages, no assistant
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "user", "content": "Another question."}
        ]
        example = format_training_example(self.tokenizer, messages)
        labels = example["labels"]
        num_unmasked = (labels != -100).sum().item()
        self.assertGreater(num_unmasked, 0, "All labels are masked even with no assistant!")

    def test_no_assistant_token_fallback(self):
        # No assistant, should fallback to last non-padding token
        messages = [
            {"role": "user", "content": "Only user here."}
        ]
        example = format_training_example(self.tokenizer, messages)
        labels = example["labels"]
        input_ids = example["input_ids"]
        # Find last non-padding token
        nonpad_indices = (input_ids != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        last_idx = nonpad_indices[-1].item() if len(nonpad_indices) > 0 else -1
        self.assertNotEqual(labels[last_idx].item(), -100, "Last non-padding token should be unmasked if no assistant.")

if __name__ == "__main__":
    unittest.main()
