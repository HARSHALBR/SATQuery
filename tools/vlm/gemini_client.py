import os
from typing import List
from pydantic import ValidationError
from google import genai
from google.genai import types

from schemas.vlm import VLMResult, VLMContext, VLMClaimType
from tools.vlm.client import VLMClient
from tools.vlm.prompt import build_vlm_prompt, VLM_SYSTEM_PROMPT

class GeminiVLMClient(VLMClient):
    """
    Real VLM provider using Google Gemini via the google-genai SDK.
    Supports gemini-3.6-flash for structured output analysis of temporal composite images.
    """
    def __init__(self, model_name: str = "gemini-3.6-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing. Cannot instantiate GeminiVLMClient.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def analyze(self, image_paths: List[str], query: str, context: VLMContext) -> VLMResult:
        if not image_paths:
            raise ValueError("GeminiVLMClient requires at least one composite image path.")
            
        composite_path = image_paths[0]
        if not os.path.exists(composite_path):
            raise FileNotFoundError(f"Image not found: {composite_path}")
            
        prompt = build_vlm_prompt(query, context.t1_date, context.t2_date)
        
        composite_guidance = (
            "\n\nIMAGE LAYOUT INSTRUCTION:\n"
            "You have been provided a single composite image. "
            "The LEFT half of this image is T1 (Before). "
            "The RIGHT half of this image is T2 (After). "
            "Evaluate changes strictly from T1 (Left) to T2 (Right)."
        )
        final_prompt = prompt + composite_guidance

        try:
            with open(composite_path, "rb") as f:
                image_bytes = f.read()
            
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[final_prompt, image_part],
                config=types.GenerateContentConfig(
                    system_instruction=VLM_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=VLMResult,
                    temperature=0.1
                )
            )
            
            response_text = response.text
            if not response_text:
                raise ValueError("Received empty response from Gemini.")
                
            return VLMResult.model_validate_json(response_text)
            
        except ValidationError as e:
            raise ValueError(f"Malformed VLM response (schema violation): {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Gemini API execution failed: {str(e)}")
