# -----------------------
# Gemini LLM
# -----------------------
import time
import google.generativeai as genai
from google.api_core import exceptions
from config import GEN_MODEL, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH
from text_embeddings import ensure_gemini

def call_llm(system_prompt: str, user_prompt: str, max_retries: int = 5, agent_name: str = "LLM") -> str:
    """
    Call the LLM with retry logic for rate limit errors.
    
    Args:
        system_prompt: System instruction for the model
        user_prompt: User prompt/query
        max_retries: Maximum number of retry attempts (default: 5)
        agent_name: Name of the agent calling this (for debugging)
    
    Returns:
        Generated text response (truncated to MAX_OUTPUT_LENGTH)
    """
    # Truncate inputs if too long
    system_len = len(system_prompt)
    user_len = len(user_prompt)
    total_input_len = system_len + user_len
    
    if user_len > MAX_INPUT_LENGTH:
        print(f"[{agent_name}] WARNING: User prompt too long ({user_len} chars), truncating to {MAX_INPUT_LENGTH} chars")
        user_prompt = user_prompt[:MAX_INPUT_LENGTH] + "\n\n[TRUNCATED...]"
        user_len = len(user_prompt)
        total_input_len = system_len + user_len
    
    print(f"[{agent_name}] Input lengths - System: {system_len:,} chars, User: {user_len:,} chars, Total: {total_input_len:,} chars")
    
    ensure_gemini()
    model = genai.GenerativeModel(GEN_MODEL, system_instruction=system_prompt)
    
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(
                user_prompt, 
                generation_config=genai.types.GenerationConfig(temperature=0.4)
            )
            output = (resp.text or "").strip()
            output_len = len(output)
            output_words = len(output.split())
            
            # Truncate output if too long
            if output_len > MAX_OUTPUT_LENGTH:
                print(f"[{agent_name}] WARNING: Output too long ({output_len:,} chars, ~{output_words:,} words), truncating to {MAX_OUTPUT_LENGTH:,} chars")
                output = output[:MAX_OUTPUT_LENGTH] + "\n\n[OUTPUT TRUNCATED...]"
                output_len = len(output)
                output_words = len(output.split())
            
            print(f"[{agent_name}] Output length: {output_len:,} chars (~{output_words:,} words)")
            return output
        
        except exceptions.ResourceExhausted as e:
            if attempt < max_retries - 1:
                # Exponential backoff: wait 2^attempt seconds
                wait_time = 2 ** attempt
                print(f"[WARNING] Rate limit hit (429). Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"[ERROR] Rate limit exceeded after {max_retries} attempts. Please wait and try again later.")
                raise
        
        except Exception as e:
            # For other errors, retry with exponential backoff
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"[WARNING] API error: {e}. Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"[ERROR] Failed after {max_retries} attempts: {e}")
                raise
    
    # Should never reach here, but just in case
    return ""
