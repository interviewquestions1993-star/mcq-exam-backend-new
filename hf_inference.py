"""
Hugging Face Inference Providers API wrapper using OpenAI-compatible endpoint
Uses the new unified endpoint: https://router.huggingface.co/v1
"""
from openai import OpenAI
from config import HF_TOKEN, HF_MODEL
import traceback
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceInference:
    """Wrapper for Hugging Face Inference Providers (OpenAI-compatible API)"""
    
    def __init__(self):
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN
        )
        self.model = HF_MODEL
        logger.info(f"DEBUG: Initialized Inference Client")
        logger.info(f"DEBUG: Model: {self.model}")
        logger.info(f"DEBUG: Token provided: {bool(HF_TOKEN)}")
    
    def text_generation(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text using Hugging Face Inference Providers
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0 = deterministic, 1.0+ = random)
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated text
            
        Raises:
            Exception: Detailed error from HF API or connection issues
        """
        try:
            logger.info(f"Sending request to HF API for model: {self.model}")
            logger.info(f"Prompt length: {len(prompt)} characters")
            logger.info(f"Parameters - max_tokens: {max_new_tokens}, temp: {temperature}, top_p: {top_p}")
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            generated_text = completion.choices[0].message.content
            logger.info(f"Successfully generated text of length: {len(generated_text)}")
            return generated_text
            
        except Exception as e:
            error_message = str(e)
            error_type = type(e).__name__
            
            logger.error(f"HF API Error Type: {error_type}")
            logger.error(f"HF API Error Message: {error_message}")
            logger.error(f"Full Traceback:\n{traceback.format_exc()}")
            
            # Return detailed error info
            raise Exception(f"HuggingFace API Error ({error_type}): {error_message}")


# Create singleton instance
hf_client = HuggingFaceInference()
