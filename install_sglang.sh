
sudo apt-get install protobuf-compiler -y
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# install nixl for cu129
pip install nixl[cu12]

# install sglang
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e "python"
