# WB-AI

WB-AI is a C++ library with a CPython extension for running an open-weight language model on your own computer. Inference uses llama.cpp and a local GGUF model file. The library does not call OpenAI or another hosted model API.

This is an early local-chat implementation, not a model trained from scratch or a replacement for a commercial chatbot. Fine-tuning starts from an existing open-weight model and uses your conversation examples.

## Build the Python package

Requirements: Python 3.9+, a C++17 compiler, CMake 3.20+, and Git. CMake fetches llama.cpp during the build, so the first build needs internet access. After the build, generation uses the local model file and does not need a network connection.

```bash
python -m pip install .
```

## Run a chat

Download a Qwen instruct GGUF model that fits your computer, for example a Qwen3 0.6B instruct GGUF. Keep the model on your machine; this library does not download model weights for you.

```python
from wb_ai import ChatBot

bot = ChatBot("models/qwen3-0.6b-instruct-q4_k_m.gguf")
print(bot.ask("Hello!"))
print(bot.ask("Can you explain that more simply?"))
```

`ChatBot` keeps the conversation messages between calls. Use `bot.clear()` to start a new conversation. For direct control, `wb_ai.Model(path).chat([("user", "Hello")])` is available. The first implementation formats prompts using the Qwen ChatML convention, so use a compatible Qwen instruct model.

The runtime currently uses CPU inference. `context_size`, `max_tokens`, and `temperature` can be set in Python:

```python
bot = ChatBot("models/model.gguf", context_size=4096)
reply = bot.ask("Tell me a short story", max_tokens=200, temperature=0.7)
```

## Fine-tune on your own examples (optional)

Training dependencies are separate from the C++ inference runtime. LoRA fine-tuning needs an existing model, your own data, and typically a CUDA-capable GPU. It does not create a capable model from scratch. Review model and dataset licenses before using them.

Create JSONL records with a `messages` array, one conversation per line:

```json
{"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"}]}
```

Install the training tools and run the example:

```bash
python -m pip install -r training/requirements.txt
python training/train_lora.py --base-model Qwen/Qwen3-0.6B --data training/data/example.jsonl --output training/output
```

Merge the adapter into the base model:

```bash
python training/merge_lora.py --base-model Qwen/Qwen3-0.6B --adapter training/output --output training/merged
```

Use llama.cpp's `convert_hf_to_gguf.py` on `training/merged` to create a GGUF model, then load that GGUF through `ChatBot`. The training and conversion steps may download model files; normal chat inference is local.

## Current limitations

- The first version supports Qwen-style ChatML prompts and GGUF weights.
- It does not train a foundation model from scratch.
- GPU inference and streaming output are not wired up yet.
- The C++ extension uses llama.cpp's current C API; use a recent CMake, compiler, and Git installation.

## License

See the repository license file when available. llama.cpp and each model have their own licenses; check them before redistribution.
