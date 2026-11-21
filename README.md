# WB-AI

WB-AI is a C++17 knowledge retrieval library. The C++ library handles knowledge storage, website extraction, training, and responses. The optional Python module is only a `ctypes` FFI client that calls the compiled C++ library; it does not implement AI behavior.

## Build

CMake and libcurl are required.

```bash
cmake -S . -B build
cmake --build build
```

## Interactive C++ example

The demo reads questions from the terminal instead of using a hardcoded prompt. Add a `knowledge.txt` file in the working directory to supply your own knowledge; if it is absent, the demo loads a small built-in sample dataset.

```bash
./build/wbai_demo
```

Type `/quit` to exit.

## Python FFI usage

Build the shared library, then use the thin Python client. The question is read from the user at runtime:

```python
from python.wb_ai import WB_AI

with WB_AI("build/libwbai.so") as ai:
    ai.add_knowledge("cpp=C++ is a general-purpose programming language.")
    ai.train_ai(20, 0.01)
    question = input("Ask WB-AI: ")
    print(ai.reply(question))
```

`crawlWebsite(url)` fetches public HTTP or HTTPS pages with libcurl. Website access depends on network availability and the website's terms of use. Extracted content is kept as local retrieval knowledge; this small library is not a large language model and does not provide ChatGPT-like generative responses.
