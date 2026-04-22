# Install triton in the root directory
sudo rm -rf /usr/local/lib/python3.11/dist-packages/triton
sudo rm -rf /usr/local/lib/python3.11/dist-packages/triton-3.5.1-py3.11-linux-x86_64.egg/triton
python -m pip install -U triton


# Download gptoss
# for huggingface download, don't use 8118 proxy!
if [ -z "$1" ]; then
  echo "Usage: $0 <HF_TOKEN>"
  exit 1
fi
export HF_TOKEN="$1"
export HF_ENDPOINT=http://huggingface-proxy-sg.byted.org
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
python3 -m pip install --user -U "huggingface_hub[cli]" "typer>=0.12"
hf download openai/gpt-oss-120b
hf download lmsys/gpt-oss-120b-bf16
