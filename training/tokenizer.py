import logging

logger = logging.getLogger("train_lora")


class FallbackTokenizer:
    """
    Self-contained character/byte fallback tokenizer for offline environments.
    """

    def __init__(self, vocab_size: int = 151674):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token = "<|pad|>"
        self.bos_token = "<|bos|>"
        self.eos_token = "<|eos|>"

    def encode(self, text, **kwargs):
        ids = [self.bos_token_id]
        ids.extend((b + 3) % (self.vocab_size - 3) for b in text.encode("utf-8"))
        ids.append(self.eos_token_id)
        return ids

    def decode(self, ids, **kwargs):
        data = bytes(
            max(0, min(255, int(i) - 3))
            for i in ids
            if int(i) >= 3
        )
        return data.decode("utf-8", errors="replace")

    def __call__(self, text, **kwargs):
        ids = self.encode(text, **kwargs)
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }


def get_tokenizer(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    vocab_size: int = 151674,
):
    """Load Hugging Face tokenizer with FallbackTokenizer if offline."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=False,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer

    except Exception as e:
        logger.info(
            f"Using FallbackTokenizer (model_id={model_id}): {e}"
        )
        return FallbackTokenizer(vocab_size=vocab_size)
