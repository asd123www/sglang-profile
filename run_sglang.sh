# Run sglang, also needs huggingface proxy, and no http proxy.
unset HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy
export HF_ENDPOINT=http://huggingface-proxy-sg.byted.org
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
sglang serve \
  --model-path lmsys/gpt-oss-120b-bf16 \
  --tp 4 \
  --reasoning-parser gpt-oss \
  --disable-custom-all-reduce \
  --trust-remote-code
