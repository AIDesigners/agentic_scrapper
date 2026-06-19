# agentic_scrapper

The minimal system for web sires scrapping and RAG retrival (of the financial documents).

## IDE Access

```bash
ssh -L 3000:ifo2:23 ayakovenko@ssh.ifowonco.com
ssh -p 3000 ayakovenko@127.0.0.1
```

## Graphic server access

```bash
sudo apt update
sudo apt install xrdp
sudo adduser xrdp ssl-cert # Gives xrdp permission to use security certificates
sudo systemctl restart xrdp
```

Open the config file: sudo nano /etc/xrdp/xrdp.ini
```
allow_channels=true
require_credentials=true
```

Add after globals right before [Xorg]:
```
 [VNC-Agent-Session] 
name=VNC-Agent-Session 
lib=libvnc.so 
username=na 
password=ask 
ip=127.0.0.1
port=5901
```

Restart: 
```bash
sudo systemctl restart xrdp
```

From windows:

```bash
ssh -J ayakovenko@ssh.ifowonco.com:22 ayakovenko@ifo2 -p 23 -L 33389:localhost:3389 
```

Win + R
"Remote Desktop Connection" (mstsc) user ayakovenko address localhost:33389


# LLM server

```bash
python3 -m vllm.entrypoints.openai.api_server --model neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16 --host 0.0.0.0 --port 8000 --api-key alex_llm_qwen --dtype auto --max-model-len 73728 --kv-cache-dtype fp8 --gpu-memory-utilization 0.95 --enable-auto-tool-choice --tool-call-parser qwen3_xml --quantization compressed-tensors
```

```bash
python3 -m vllm.entrypoints.openai.api_server --model QuantTrio/Qwen3.6-9B-AWQ --host 0.0.0.0 --port 8000 --api-key alex_llm_qwen --max-model-len 131072 --kv-cache-dtype auto--gpu-memory-utilization 0.92 --enable-auto-tool-choice  --reasoning-parser qwen3  --tool-call-parser qwen3_xml --enable-prefix-caching 
```


Sample query:
```bash
curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{"messages": [{"role": "user", "content": "What is the capital of US"}],  "max_tokens": 113584}'
curl ifo4:8000/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer alex_llm_qwen" -d '{"model": "neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16", "messages": [{"role": "user", "content": "What is the speed of light in meters per second?"}]}'
```
