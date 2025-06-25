#!/bin/bash

pip install -e .

sed -i '1161s/^[ \t]*logits = logits.float()/#&/' /opt/conda/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py
sed -i '1044s/^[ \t]*logits = logits.float()/#&/' /opt/conda/lib/python3.10/site-packages/transformers/models/mistral/modeling_mistral.py
sed -i '1069s/^[ \t]*logits = logits.float()/#&/' /opt/conda/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py