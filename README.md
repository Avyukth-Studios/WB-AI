# WB-AI

WB-AI is a small C++ knowledge library with a C-compatible Python FFI.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## C++ example

```cpp
#include "WB.h"
#include <iostream>

int main() {
    Mini_AI ai;
    ai.addKnowledge("cpp=C++ is a general-purpose programming language.");
    ai.train_ai(20, 0.01);
    std::cout << ai.aiReply("What is cpp?") << '\n';
}
```

## Python example

```python
import ctypes

library = ctypes.CDLL("./build/libwbai.so")
library.ai_create.restype = ctypes.c_void_p
library.ai_reply.restype = ctypes.c_char_p
library.ai_reply.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

ai = library.ai_create()
library.ai_add_knowledge(ai, b"cpp=C++ is a general-purpose programming language.")
library.ai_train(ai, 20, 0.01)
print(library.ai_reply(ai, b"What is cpp?").decode())
library.ai_destroy(ai)
```
