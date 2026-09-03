"""
RS-InternVL: Dual-Branch Remote Sensing Vision-Language Model.

Connects Sentinel-1 SAR and Sentinel-2 Multispectral encoders to an authentic
InternVL3-1B-class language backbone through non-linear modality projections
and multimodal token fusion.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from models.rs_internvl.config import RSInternVLConfig
from models.rs_internvl.fusion import FusedMultimodalOutput, MultimodalTokenFusion
from models.rs_internvl.projection import S1Projection, S2Projection
from models.rs_internvl.s1_encoder import S1Encoder
from models.rs_internvl.s2_encoder import S2Encoder

logger = logging.getLogger(__name__)


class RSInternVL(nn.Module):
    """
    RS-InternVL Model Architecture.
    
    Structure:
    - Sentinel-1 SAR Branch: S1Encoder (2-channel) -> S1Projection -> LLM Space
    - Sentinel-2 MS Branch: S2Encoder (10-channel) -> S2Projection -> LLM Space
    - Multimodal Token Fusion: Assembles visual tokens and text into inputs_embeds sequence
    - Language Model Backbone: Authentic InternVL3-1B / Qwen2 Causal Language Model
    """

    def __init__(self, config: Optional[RSInternVLConfig] = None):
        super().__init__()
        self.config = config if config is not None else RSInternVLConfig()

        # 1. Modality Encoders (Task-specific initialization, frozen initially for Step 2)
        self.s1_encoder = S1Encoder(
            in_channels=self.config.s1_channels,
            hidden_dim=self.config.s1_hidden_dim,
            freeze_backbone=self.config.freeze_s1_encoder,
        )
        self.s2_encoder = S2Encoder(
            in_channels=self.config.s2_channels,
            hidden_dim=self.config.s2_hidden_dim,
            img_size=self.config.img_size,
            freeze_backbone=self.config.freeze_s2_encoder,
        )

        # 2. Modality Projections (Trainable)
        self.s1_projection = S1Projection(
            in_dim=self.config.s1_hidden_dim,
            out_dim=self.config.llm_hidden_dim,
            hidden_dim=self.config.projection_hidden_dim,
            dropout=self.config.projection_dropout,
        )
        self.s2_projection = S2Projection(
            in_dim=self.config.s2_hidden_dim,
            out_dim=self.config.llm_hidden_dim,
            hidden_dim=self.config.projection_hidden_dim,
            dropout=self.config.projection_dropout,
        )

        # 3. Multimodal Token Fusion
        self.fusion = MultimodalTokenFusion(hidden_dim=self.config.llm_hidden_dim)

        # 4. Language Model Backbone (InternVL3-1B underlying Qwen2 architecture)
        llm_config = Qwen2Config(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.llm_hidden_dim,
            intermediate_size=self.config.intermediate_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            max_position_embeddings=self.config.max_position_embeddings,
            bos_token_id=self.config.bos_token_id,
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
        )
        self.language_model = Qwen2ForCausalLM(llm_config)

        # 5. Load authentic pretrained weights if requested
        if getattr(self.config, "pretrained_backbone", True):
            self._load_pretrained_language_model()

        if self.config.freeze_llm:
            self._freeze_llm()

    def _load_pretrained_language_model(self) -> None:
        """
        Load authentic pretrained language weights from OpenGVLab/InternVL3-1B safetensors.
        """
        if not getattr(self.config, "pretrained_backbone", True):
            return

        # If a lightweight toy config is passed for unit tests (e.g. vocab_size=1000 or num_layers=2), skip
        if (
            self.config.vocab_size != 151674
            or self.config.num_hidden_layers != 24
            or self.config.intermediate_size != 4864
        ):
            logger.info(
                "Lightweight test configuration detected (layers=%d, vocab=%d). "
                "Skipping pretrained weight loading for custom test configuration.",
                self.config.num_hidden_layers,
                self.config.vocab_size,
            )
            return

        from pathlib import Path
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except ImportError:
            logger.warning("huggingface_hub or safetensors not available. Skipping pretrained backbone loading.")
            return

        model_path = getattr(self.config, "pretrained_model_path", None)
        if model_path is None or not Path(model_path).exists():
            try:
                model_path = hf_hub_download(
                    repo_id=self.config.model_id,
                    filename="model.safetensors",
                )
            except Exception as e:
                logger.warning(f"Could not download model.safetensors for {self.config.model_id}: {e}")
                return

        try:
            logger.info(f"Loading authentic pretrained language model weights from: {model_path}")
            weights = load_file(str(model_path))
            lm_state_dict = {}
            for k, v in weights.items():
                if k.startswith("language_model."):
                    new_k = k[len("language_model."):]
                    lm_state_dict[new_k] = v

            if not lm_state_dict:
                logger.warning(f"No 'language_model.' keys found in {model_path}.")
                return

            missing, unexpected = self.language_model.load_state_dict(lm_state_dict, strict=False)
            logger.info(
                f"Successfully loaded {len(lm_state_dict)} pretrained language weights into Qwen2 backbone. "
                f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
            )
        except Exception as e:
            logger.error(f"Failed to load pretrained language weights: {e}")
            raise e

    def _freeze_llm(self) -> None:
        """Freeze language model backbone for initial projection training."""
        for param in self.language_model.parameters():
            param.requires_grad = False

    def unfreeze_llm(self) -> None:
        """Unfreeze language model backbone for full fine-tuning."""
        for param in self.language_model.parameters():
            param.requires_grad = True

    def get_num_parameters(self) -> Dict[str, int]:
        """Return parameter count breakdown across model components."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        s1_params = sum(p.numel() for p in self.s1_encoder.parameters())
        s2_params = sum(p.numel() for p in self.s2_encoder.parameters())
        proj_params = sum(p.numel() for p in self.s1_projection.parameters()) + sum(
            p.numel() for p in self.s2_projection.parameters()
        )
        llm_params = sum(p.numel() for p in self.language_model.parameters())

        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": frozen_params,
            "s1_encoder": s1_params,
            "s2_encoder": s2_params,
            "projections": proj_params,
            "language_model": llm_params,
        }

    def encode_vision(
        self,
        image_s1: torch.Tensor,
        image_s2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode and project raw multi-sensor imagery into LLM token space.
        
        Args:
            image_s1: SAR tensor [B, 2, H, W]
            image_s2: Optical tensor [B, 10, H, W]
            
        Returns:
            Tuple of (s1_features, s2_features, s1_projected_tokens, s2_projected_tokens)
        """
        # ── Band-count validation (fail-fast with actionable message) ──────────────
        expected_s1 = self.config.s1_channels
        if image_s1.shape[1] != expected_s1:
            raise ValueError(
                f"encode_vision: S1 channel mismatch. "
                f"Expected {expected_s1} channels {self.config.s1_bands}, "
                f"got {image_s1.shape[1]}. "
                f"Hint: dataset must use s1_bands={self.config.s1_bands}."
            )

        expected_s2 = self.config.s2_channels
        if image_s2.shape[1] != expected_s2:
            raise ValueError(
                f"encode_vision: S2 channel mismatch. "
                f"Expected {expected_s2} channels corresponding to model bands "
                f"{self.config.s2_bands}, but received {image_s2.shape[1]} channels. "
                f"Hint: the raw BigEarthNet S2 patch has 12 bands (B01\u2013B12 including "
                f"B01 and B09). The model requires only the 10 bands "
                f"[B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]. "
                f"Ensure the dataset is initialised with s2_bands=None (default) or "
                f"s2_bands='S2-10m20m'."
            )
        # ─────────────────────────────────────────────────────────────────────────

        # S1 SAR encoding & projection
        s1_features = self.s1_encoder(image_s1)          # [B, N_s1, s1_hidden_dim]
        s1_tokens = self.s1_projection(s1_features)      # [B, N_s1, llm_hidden_dim]

        # S2 MS encoding & projection
        s2_features = self.s2_encoder(image_s2)          # [B, N_s2, s2_hidden_dim]
        s2_tokens = self.s2_projection(s2_features)      # [B, N_s2, llm_hidden_dim]

        return s1_features, s2_features, s1_tokens, s2_tokens

    def forward(
        self,
        image_s1: torch.Tensor,
        image_s2: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        input_text: Optional[Union[str, List[str]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for RS-InternVL.
        
        Args:
            image_s1: Sentinel-1 SAR input [B, 2, H, W].
            image_s2: Sentinel-2 Multispectral input [B, 10, H, W].
            input_ids: Optional tokenized text prompt [B, L_text].
            input_text: Optional raw text prompt (used if input_ids is None).
            attention_mask: Optional text attention mask [B, L_text].
            labels: Optional language model ground-truth target token IDs [B, L_text].
            position_ids: Optional position IDs for text.
            
        Returns:
            Dictionary containing:
            - logits: Output prediction logits [B, Total_Seq_Len, vocab_size]
            - loss: Optional loss tensor (if labels provided)
            - s1_features: Raw S1 encoder features [B, N_s1, 512]
            - s2_features: Raw S2 encoder features [B, N_s2, 768]
            - s1_projected: S1 visual tokens in LLM space [B, N_s1, llm_hidden_dim]
            - s2_projected: S2 visual tokens in LLM space [B, N_s2, llm_hidden_dim]
            - fused_features: Fused multimodal input embeddings [B, Total_Seq_Len, llm_hidden_dim]
        """
        B = image_s1.shape[0]
        device = image_s1.device

        # 1. Multi-modal Vision Encoding & Projection
        s1_feat, s2_feat, s1_tok, s2_tok = self.encode_vision(image_s1, image_s2)

        # 2. Text Embedding Extraction
        if input_ids is None:
            # Default to a single BOS token if no text input_ids provided
            input_ids = torch.full(
                (B, 1), self.config.bos_token_id, dtype=torch.long, device=device
            )

        text_embeds = self.language_model.get_input_embeddings()(input_ids)  # [B, L_text, llm_hidden_dim]

        # 3. Multimodal Token Fusion with Attention Mask & Label Masking
        fused: FusedMultimodalOutput = self.fusion(
            s1_tokens=s1_tok,
            s2_tokens=s2_tok,
            text_embeds=text_embeds,
            text_attention_mask=attention_mask,
            text_labels=labels,
            text_position_ids=position_ids,
        )

        # 4. InternVL Language Model Forward Pass
        lm_outputs = self.language_model(
            inputs_embeds=fused.inputs_embeds,
            attention_mask=fused.attention_mask,
            position_ids=fused.position_ids,
            labels=fused.labels,
            return_dict=True,
        )

        return {
            "logits": lm_outputs.logits,
            "loss": getattr(lm_outputs, "loss", None),
            "s1_features": s1_feat,
            "s2_features": s2_feat,
            "s1_projected": s1_tok,
            "s2_projected": s2_tok,
            "fused_features": fused.inputs_embeds,
            "n_visual_tokens": fused.n_visual_tokens,
        }

    @torch.no_grad()
    def generate(
        self,
        image_s1: torch.Tensor,
        image_s2: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Multimodal autoregressive text generation given Sentinel-1 and Sentinel-2 inputs.
        
        Args:
            image_s1: SAR input tensor [B, 2, H, W] or [2, H, W]
            image_s2: Optical input tensor [B, 10, H, W] or [10, H, W]
            input_ids: Optional prompt token IDs [B, L_text]
            attention_mask: Optional prompt attention mask [B, L_text]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or use greedy decoding
            eos_token_id: End of sequence token ID
            
        Returns:
            Tuple of (generated_token_ids [B, L_gen], token_probabilities list)
        """
        if image_s1.ndim == 3:
            image_s1 = image_s1.unsqueeze(0)
        if image_s2.ndim == 3:
            image_s2 = image_s2.unsqueeze(0)

        B = image_s1.shape[0]
        device = image_s1.device

        # 1. Vision encoding and projection
        s1_feat, s2_feat, s1_tok, s2_tok = self.encode_vision(image_s1, image_s2)

        # 2. Text embedding
        if input_ids is None:
            input_ids = torch.full((B, 1), self.config.bos_token_id, dtype=torch.long, device=device)
        else:
            input_ids = input_ids.to(device)

        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # 3. Multimodal fusion
        fused = self.fusion(
            s1_tokens=s1_tok,
            s2_tokens=s2_tok,
            text_embeds=text_embeds,
            text_attention_mask=attention_mask,
        )

        cur_embeds = fused.inputs_embeds
        cur_mask = fused.attention_mask
        eos_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        generated_ids: List[torch.Tensor] = []
        token_probs: List[float] = []
        past_key_values = None

        # 4. Autoregressive token generation loop with KV caching
        for step_idx in range(max_new_tokens):
            outputs = self.language_model(
                inputs_embeds=cur_embeds,
                attention_mask=cur_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]  # [B, vocab_size]

            if do_sample:
                if temperature > 0:
                    scaled_logits = next_token_logits / temperature
                else:
                    scaled_logits = next_token_logits
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                prob_val = probs[0, next_token[0, 0]].item()
            else:
                probs = torch.softmax(next_token_logits, dim=-1)
                prob_val, next_token = torch.max(probs, dim=-1, keepdim=True)
                prob_val = prob_val[0, 0].item()

            generated_ids.append(next_token)
            token_probs.append(prob_val)

            if next_token[0, 0].item() == eos_id:
                break

            # Forward only the newly generated single token at subsequent steps
            cur_embeds = self.language_model.get_input_embeddings()(next_token)
            if cur_mask is not None:
                next_mask = torch.ones((B, 1), dtype=cur_mask.dtype, device=device)
                cur_mask = torch.cat([cur_mask, next_mask], dim=1)

        if generated_ids:
            gen_tensor = torch.cat(generated_ids, dim=1)
        else:
            gen_tensor = torch.empty((B, 0), dtype=torch.long, device=device)

        return gen_tensor, token_probs

    @torch.no_grad()
    def predict(
        self,
        image_s1: torch.Tensor,
        image_s2: torch.Tensor,
        query: str,
        tokenizer: Optional[Any] = None,
        max_new_tokens: int = 64,
        do_sample: bool = False,
    ) -> Dict[str, Any]:
        """
        Structured inference interface performing actual multimodal generation.
        
        Args:
            image_s1: Sentinel-1 SAR tensor [B, 2, H, W] or [2, H, W].
            image_s2: Sentinel-2 MS tensor [B, 10, H, W] or [10, H, W].
            query: Natural language query string.
            tokenizer: Tokenizer instance.
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to sample or use greedy generation.
            
        Returns:
            Structured dictionary:
            {
                "answer": str,
                "claim": str,
                "claim_type": str,
                "model_score": float,
                "model_version": str,
                "grounding": Optional[Any]
            }
        """
        if image_s1.ndim == 3:
            image_s1 = image_s1.unsqueeze(0)
        if image_s2.ndim == 3:
            image_s2 = image_s2.unsqueeze(0)

        # 1. Format chat prompt
        prompt_str = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

        # 2. Tokenize prompt if tokenizer provided
        input_ids = None
        attention_mask = None

        if tokenizer is not None:
            if hasattr(tokenizer, "encode"):
                tok_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
                input_ids = torch.tensor([tok_ids], dtype=torch.long, device=image_s1.device)
                attention_mask = torch.ones_like(input_ids)
            else:
                enc = tokenizer(prompt_str, return_tensors="pt")
                input_ids = enc["input_ids"].to(image_s1.device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(image_s1.device)

        # 3. Generate tokens
        gen_tokens, token_probs = self.generate(
            image_s1=image_s1,
            image_s2=image_s2,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

        # 4. Decode generated tokens
        if tokenizer is not None and hasattr(tokenizer, "decode") and gen_tokens.numel() > 0:
            token_list = gen_tokens[0].tolist()
            raw_text = tokenizer.decode(token_list)
            answer_text = raw_text.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("<|endoftext|>", "").strip()
            if not answer_text:
                answer_text = raw_text.strip()
        else:
            answer_text = f"Tokens: {gen_tokens[0].tolist() if gen_tokens.numel() > 0 else []}"

        # 5. Compute real model confidence score
        if token_probs:
            model_score = round(float(sum(token_probs) / len(token_probs)), 4)
        else:
            forward_out = self.forward(image_s1=image_s1, image_s2=image_s2, input_ids=input_ids)
            probs = torch.softmax(forward_out["logits"][:, -1, :], dim=-1)
            model_score = round(torch.max(probs, dim=-1)[0].item(), 4)

        # 6. Infer claim type
        q_lower = query.lower()
        if any(q_lower.startswith(w) for w in ("is ", "are ", "does ", "can ", "do ")):
            claim_type = "presence_verification"
        elif any(w in q_lower for w in ("what ", "which ", "dominant", "classify")):
            claim_type = "land_cover_classification"
        else:
            claim_type = "visual_question_answering"

        return {
            "answer": answer_text,
            "claim": f"Multi-modal SAR (VV/VH) and Optical (10 bands) query: {query}",
            "claim_type": claim_type,
            "model_score": model_score,
            "model_version": f"RS-InternVL3-1B-LoRA (backbone: {self.config.model_id})",
            "grounding": None,
        }
