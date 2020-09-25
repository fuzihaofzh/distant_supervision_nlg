g++ -std=c++11 -pthread -O3 tools/fastBPE/fastBPE/main.cc -IfastBPE -o tools/fastBPE/fast
pip3 install -r requirements.txt
pip3 install -r e2e-metrics/requirements.txt
pip3 install --editable fairseq/