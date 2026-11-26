#   --------   L L M   C O N F I G   --------
#
# The LLM connection and model constants shared by every agent, configured
# in ONE place: the base vLLM endpoint (the big base LLM - the story
# extraction / the crawlers), the perplexity servitor (a llama.cpp wrapper
# running a small model on a dedicated host), and the jina-v3 embedding
# servitor (a dedicated vLLM instance). Each agent imports its endpoint's
# block from this module.

import os


#  ------------   B A S E   L L M   E N D P O I N T   ( v L L M ,   p o r t   8 0 0 0 )   ------------ #
BASE_URL = "http://ifo4:8000"
BASE_API_KEY = "alex_llm_qwen"
MODEL_NAME = "unsloth/Qwen3.8-27B-NVFP4"
# The servitor's launch command:
#python3  -m vllm.entrypoints.openai.api_server --model unsloth/Qwen3.8-27B-NVFP4 --host 0.0.0.0 --port 8000 --api-key alex_llm_qwen \
#        --max-model-len 163840 --kv-cache-dtype fp8 --gpu-memory-utilization 0.96 --max-num-seqs 1 --max-num-batched-tokens 8192    \
#        --enable-auto-tool-choice --tool-call-parser qwen3_xml --enable-prefix-caching --reasoning-parser qwen3 --trust-remote-code \
#        --enable-chunked-prefill --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}'
# The shared context/output budgets of the base LLM (the Yahoo agent keeps
# its own yahoo-specific pair in crawler_agent_yahoo.py).
LLM_BASE_CONTEXT_LEN = 163840 # 131072 # 196608
LLM_MAX_OUTPUT_TOKENS = 32768
# --- LLM liveness tuning (env-overridable) ---
# Busy vLLM (max-num-seqs 1) queues a request behind the in-flight
# generation, so a single call can legitimately take a long time in a slow
# environment: the read timeout of the LLM client is capped at 2h by default
# (LLM_READ_TIMEOUT, seconds) - long enough for busy servers, short enough
# to bound a wedged one - and the SDK-level retries are disabled
# (max_retries=0) so the harness attempt budget is the single retry
# authority. A truly dead vLLM peer is surfaced by the default TCP keepalive
# (langchain-openai) in about 2 minutes, independent of the read timeout.
LLM_READ_TIMEOUT_S = float(os.environ.get("LLM_READ_TIMEOUT", "7200"))
# The startup health gate sends a trivial "ping" generation, which on a busy
# server queues behind the in-flight generation: give it 600s by default
# (LLM_PING_TIMEOUT, seconds) instead of failing a healthy-but-busy server.
LLM_PING_TIMEOUT_S = float(os.environ.get("LLM_PING_TIMEOUT", "600"))


#  ------------   P E R P L E X I T Y   S E R V I T O R   ( l l a m a . c p p ,   p o r t   8 0 0 2 )   ------------ #
# The perplexity servitor (a llama.cpp wrapper running a small model on
# a dedicated host): the post-accept perplexity of a story is measured
# against this model, not against the big base LLM.
SERVITOR_BASE_URL = "http://ifo4:8002"
SERVITOR_API_KEY = "alex_llm_qwen"
SERVITOR_MODEL_NAME = "qwen35-4b"
# The post-accept perplexity measurement (monitoring only) is a single
# prompt-scoring pass over the whole story on the servitor's dedicated
# host: 300s bounds a wedged servitor without queueing behind long
# extraction generations on the base LLM.
PERPLEXITY_TIMEOUT_S = 300.0


#  ------------   E M B E D D I N G   S E R V I T O R   ( v L L M ,   p o r t   8 0 0 1 )   ------------ #
# The jina-v3 embedding servitor (a dedicated vLLM instance) shared by the
# RAG ingestion agent (ingester_rag.py): the servitor exposes the
# OpenAI-compatible /tokenize, /detokenize and /v1/embeddings endpoints -
# all the embedding model work (the story tokenization, the equal-size
# windowing in the model's token space, the precomputed EMBEDDING_DIM-dim
# chunk vectors) happens against this endpoint over HTTP.
EMBEDDING_BASE_URL = "http://ifo4:8001"
EMBEDDING_API_KEY = "alex_llm_qwen"
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v3"
EMBEDDING_DIM = 1024  # the model's embedding dimension (probe-confirmed)
# The servitor's launch command (a CPU deployment - the first ~8k-token
# forward absorbs a one-time inductor compilation, so the per-step RPC
# deadline must cover it: VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600).
#export VLLM_TARGET_DEVICE="cpu" VLLM_CPU_KVCACHE_SPACE=2 VLLM_CPU_OMP_THREADS_BIND="4-5" VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600
#python3  -m vllm.entrypoints.openai.api_server --model jinaai/jina-embeddings-v3 --host 0.0.0.0 --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.24 --api-key alex_llm_qwen --runner pooling --convert embed --trust-remote-code
