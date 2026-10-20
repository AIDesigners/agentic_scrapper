#!/usr/bin/env python3
import argparse
import math
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from llama_cpp import Llama
from pydantic import BaseModel

# 1. Parse Command-Line Arguments
parser = argparse.ArgumentParser(description="Secure llama.cpp Perplexity & Completion Server")
parser.add_argument("--model",   type=str, default="/home/ayakovenko/llama-env-cpu/models/Qwen3.5-4B-MTP-UD-Q8_K_XL.gguf", help="Path to GGUF model file")
parser.add_argument("--host",    type=str, default="0.0.0.0",       help="Bind host")
parser.add_argument("--port",    type=int, default=8002,            help="Bind port")
parser.add_argument("--api-key", type=str, default="alex_llm_qwen", help="Secret API key for Bearer auth")
# NOTE: the logits_all scores buffer is sized n_ctx * n_vocab * 4 bytes
# (~8 MB per 1k ctx tokens for a 248k-vocab model). Keep the default
# modest: 8192 ctx ~= 8 GB of buffer on top of the model weights.
parser.add_argument("--ctx",     type=int, default=8192,            help="Context window length")
parser.add_argument("--ncpus",   type=int, default=4,               help="Number of CPU threads")
args = parser.parse_args()

# 2. Initialize FastAPI and Security Scheme
app = FastAPI(title="Llama.cpp Perplexity Wrapper", version="1.0")
security = HTTPBearer()

# 3. Load the Llama Model into Memory (with logits enabled for logprob extraction)
print(f"Loading model {args.model} with {args.ncpus} threads...")
llm = Llama(
    model_path=args.model,
    n_ctx=args.ctx,
    n_threads=args.ncpus,
    logits_all=True,
    verbose=False
)
n_vocab = llm.n_vocab()
print("Model loaded successfully.")
print(f"Context: {args.ctx} tokens, vocab: {n_vocab}, "
      f"scores buffer: {args.ctx * n_vocab * 4 / 2**30:.1f} GiB")

# 4. Authentication Dependency
def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != args.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# 5. Request/Response Schemas (OpenAI-compatible subset)
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 1
    prompt_logprobs: int = 0

@app.post("/v1/llama-perplexity")
async def create_completion(req: CompletionRequest, token: str = Security(verify_api_key)):
    try:
        # Tokenize input prompt
        input_ids = llm.tokenize(req.prompt.encode("utf-8"))

        # The prompt must fit in the context window (plus room for one score row)
        if len(input_ids) > args.ctx:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Prompt tokenizes to {len(input_ids)} tokens, exceeding context length {args.ctx}. "
                       f"Send a shorter prompt or restart the server with a larger --ctx "
                       f"(note: the scores buffer costs ~{n_vocab * 4 / 1024:.0f} KiB per context token).",
            )

        # Reset and evaluate prompt tokens to populate logits
        llm.reset()
        llm.eval(input_ids)

        # Get the valid logits slice: shape [n_tokens, n_vocab].
        # Row i holds the logits for predicting token i+1 given tokens 0..i.
        logits = llm._scores[: len(input_ids), :]

        # Vectorized log-softmax over the whole matrix (numpy, not per-element Python).
        log_probs = Llama.logits_to_logprobs(logits)

        # Logprob of token i (i >= 1) is log_probs[i-1, input_ids[i]]:
        # row r holds the logits that predict the token at position r+1.
        # The first token has no prior context, so it gets no logprob.
        prompt_logprobs_list = [None]
        if len(input_ids) > 1:
            rows = np.arange(len(input_ids) - 1)
            cols = np.asarray(input_ids[1:], dtype=np.intp)
            token_lps = log_probs[rows, cols]

            for token_id, token_lp in zip(input_ids[1:], token_lps):
                token_str = llm.detokenize([int(token_id)]).decode("utf-8", errors="ignore")
                prompt_logprobs_list.append({
                    token_str: {
                        "logprob": float(token_lp),
                        "bytes": list(token_str.encode("utf-8"))
                    }
                })

            # Calculate overall sequence perplexity from valid logprobs
            perplexity = math.exp(-float(np.mean(token_lps)))
        else:
            perplexity = 0.0

        return {
            "id": "cmpl-llama-cpp",
            "object": "text_completion",
            "choices": [
                {
                    "text": "",
                    "index": 0,
                    "prompt_logprobs": prompt_logprobs_list,
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": len(input_ids),
                "completion_tokens": 0,
                "total_tokens": len(input_ids)
            },
            "perplexity": perplexity
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)

