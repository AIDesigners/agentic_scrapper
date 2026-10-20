#   --------   A G E N T I C   T O O L S   --------
#
# Generic agentic (LLM-side) tools shared by every crawler agent:
# the vLLM endpoint constants and the tokenizer helpers (token counting
# + prompt truncation) that talk to the OpenAI-compatible /tokenize and
# /detokenize endpoints. Nothing web/browser-related lives here - that
# belongs in web_tools.py.

import httpx


#  ------------   L L M   E N D P O I N T   C O N S T A N T S   ------------ #
BASE_URL="http://ifo4:8000"
BASE_API_KEY="alex_llm_qwen"
#MODEL_NAME="poolside/Laguna-XS-2.1-NVFP4"
#MODEL_NAME="QuantTrio/Qwen3.6-27B-AWQ"
MODEL_NAME="unsloth/Qwen3.8-27B-NVFP4"
#python3  -m vllm.entrypoints.openai.api_server --model unsloth/Qwen3.8-27B-NVFP4 --host 0.0.0.0 --port 8000 --api-key alex_llm_qwen \
#        --max-model-len 163840 --kv-cache-dtype fp8 --gpu-memory-utilization 0.96 --max-num-seqs 1 --max-num-batched-tokens 8192    \
#        --enable-auto-tool-choice --tool-call-parser qwen3_xml --enable-prefix-caching --reasoning-parser qwen3 --trust-remote-code \
#        --enable-chunked-prefill --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}'


#  ------------   T O K E N I Z E R   H E L P E R S   ------------ #

# This routine extract amount of tokens 'locked' in tools
async def get_tools_tokens_length(tools_str: str) -> int :
    headers = {"Authorization": f"Bearer {BASE_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers, timeout=60.0) as client :
        response = await client.post( "/tokenize", json={ "model": MODEL_NAME, "prompt": tools_str, } )
        response.raise_for_status()
    return response.json().get("count", len(response.json().get("tokens", [])))
# This routine truncates the prompt to fit into a model
async def truncate_prompt(prompt: str, max_tokens_number: int) -> str :
    headers={ "Authorization": f"Bearer {BASE_API_KEY}", "Content-Type": "application/json" }
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers, timeout=30.0) as client:
        response = await client.post("/tokenize", json={ "model": MODEL_NAME, "prompt": prompt, })
        response.raise_for_status()
        tokens = response.json().get("tokens", [])
        if response.json().get("count", len(tokens)) > max_tokens_number :
            response = await client.post("/detokenize", json={"model": MODEL_NAME, "tokens": tokens[: max_tokens_number]})
            response.raise_for_status()
            return response.json().get("text", "")
        else :
            return prompt
