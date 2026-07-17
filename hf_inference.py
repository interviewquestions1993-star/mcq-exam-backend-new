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
            logger.info(f"Sending request to AI API for model: {self.model}")
            logger.info(f"Prompt length: {len(prompt)} characters")
            logger.info(f"Parameters - max_tokens: {max_new_tokens}, temp: {temperature}, top_p: {top_p}")
            
            # NVIDIA client initialization
            from config import NVIDIA_API_KEY
            if not NVIDIA_API_KEY:
                logger.warning("NVIDIA_API_KEY is missing. AI requests may fail.")
            
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=NVIDIA_API_KEY or "dummy_key"
            )
            
            models_to_try = [
                "nvidia/nemotron-3-ultra-550b-a55b",
                "meta/llama3-70b-instruct",
                "bytedance/seed-oss-36b-instruct",
                "meta/llama-3.2-3b-instruct",
                "google/gemma-4-31b-it",
                "meta/llama-3.2-1b-instruct",
                "meta/llama-3.1-8b-instruct",
                "minimaxai/minimax-m3",
                "mistralai/mistral-small-4-119b-2603",
                "nvidia/nemotron-3-super-120b-a12b",
                "qwen/qwen3.5-122b-a10b"
            ]
            
            last_error = None
            for current_model in models_to_try:
                try:
                    logger.info(f"Attempting with model: {current_model}")
                    completion = client.chat.completions.create(
                        model=current_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_new_tokens,
                        extra_body={"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 16384},
                        stream=True,
                        timeout=20
                    )
                    
                    # Accumulate the final text content
                    final_content = ""
                    reasoning_content = ""
                    
                    for chunk in completion:
                        if not chunk.choices:
                            continue
                        reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                        if reasoning:
                            reasoning_content += reasoning
                            
                        if chunk.choices[0].delta.content is not None:
                            final_content += chunk.choices[0].delta.content
                            
                    logger.info(f"Successfully generated text of length: {len(final_content)}")
                    return final_content
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"Model {current_model} failed: {str(e)}")
                    continue
                    
            # If all models failed
            error_message = str(last_error)
            error_type = type(last_error).__name__
            logger.error(f"AI API Error Type: {error_type}")
            logger.error(f"AI API Error Message: {error_message}")
            logger.error(f"Full Traceback:\n{traceback.format_exc()}")
            raise Exception(f"AI API Error after fallbacks ({error_type}): {error_message}")
            
        except Exception as e:
            raise e


# Create singleton instance
hf_client = HuggingFaceInference()
