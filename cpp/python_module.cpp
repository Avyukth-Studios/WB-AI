#include <Python.h>
#include "llama.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

struct WBModel {
    PyObject_HEAD
    llama_model *model;
    llama_context *context;
    int context_size;
};

static PyObject *WBModel_new(PyTypeObject *type, PyObject *, PyObject *)
{
    WBModel *self = (WBModel *)type->tp_alloc(type, 0);
    if (self != nullptr) {
        self->model = nullptr;
        self->context = nullptr;
        self->context_size = 4096;
    }
    return (PyObject *)self;
}

static int WBModel_init(WBModel *self, PyObject *args, PyObject *kwargs)
{
    const char *path = nullptr;
    int context_size = 4096;
    static char *names[] = {
        (char *)"model_path",
        (char *)"context_size",
        nullptr
    };

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "s|i", names, &path, &context_size)) {
        return -1;
    }

    if (context_size < 128) {
        PyErr_SetString(PyExc_ValueError, "context_size must be at least 128");
        return -1;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    llama_model *model = llama_model_load_from_file(path, model_params);
    if (model == nullptr) {
        PyErr_Format(PyExc_RuntimeError, "Could not load GGUF model: %s", path);
        return -1;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = (uint32_t)context_size;

    llama_context *context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        llama_model_free(model);
        PyErr_SetString(PyExc_RuntimeError, "Could not create model context");
        return -1;
    }

    if (self->context != nullptr) {
        llama_free(self->context);
    }
    if (self->model != nullptr) {
        llama_model_free(self->model);
    }

    self->model = model;
    self->context = context;
    self->context_size = context_size;
    return 0;
}

static void WBModel_dealloc(WBModel *self)
{
    if (self->context != nullptr) {
        llama_free(self->context);
    }
    if (self->model != nullptr) {
        llama_model_free(self->model);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static std::string make_prompt(PyObject *messages)
{
    PyObject *sequence = PySequence_Fast(
        messages, "messages must be a sequence of (role, content) pairs");
    if (sequence == nullptr) {
        throw std::runtime_error("messages must be a sequence of pairs");
    }

    std::string prompt;
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence);
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject *pair = PySequence_Fast_GET_ITEM(sequence, i);
        PyObject *pair_sequence = PySequence_Fast(
            pair, "each message must be a (role, content) pair");
        if (pair_sequence == nullptr) {
            Py_DECREF(sequence);
            throw std::runtime_error("each message must be a (role, content) pair");
        }

        if (PySequence_Fast_GET_SIZE(pair_sequence) != 2) {
            Py_DECREF(pair_sequence);
            Py_DECREF(sequence);
            throw std::runtime_error("each message must contain a role and content");
        }

        PyObject *role_object = PySequence_Fast_GET_ITEM(pair_sequence, 0);
        PyObject *text_object = PySequence_Fast_GET_ITEM(pair_sequence, 1);
        const char *role = PyUnicode_AsUTF8(role_object);
        const char *text = PyUnicode_AsUTF8(text_object);
        if (role == nullptr || text == nullptr) {
            Py_DECREF(pair_sequence);
            Py_DECREF(sequence);
            throw std::runtime_error("message role and content must be strings");
        }

        const std::string role_string(role);
        if (role_string != "system" && role_string != "user" &&
            role_string != "assistant") {
            Py_DECREF(pair_sequence);
            Py_DECREF(sequence);
            throw std::runtime_error("role must be system, user, or assistant");
        }

        prompt += "<|im_start|>";
        prompt += role;
        prompt += "\n";
        prompt += text;
        prompt += "<|im_end|>\n";
        Py_DECREF(pair_sequence);
    }
    Py_DECREF(sequence);

    prompt += "<|im_start|>assistant\n";
    return prompt;
}

static std::string generate_reply(
    WBModel *self,
    const std::string &prompt,
    int max_tokens,
    float temperature)
{
    llama_context *context = self->context;
    const llama_model *model = self->model;
    const llama_vocab *vocab = llama_model_get_vocab(model);

    llama_memory_clear(llama_get_memory(context), true);

    int token_count = llama_tokenize(
        vocab, prompt.c_str(), (int32_t)prompt.size(), nullptr, 0, true, true);
    if (token_count >= 0) {
        throw std::runtime_error("Could not calculate prompt token count");
    }
    token_count = -token_count;

    std::vector<llama_token> tokens((size_t)token_count);
    int actual_count = llama_tokenize(
        vocab, prompt.c_str(), (int32_t)prompt.size(),
        tokens.data(), token_count, true, true);
    if (actual_count < 0) {
        throw std::runtime_error("Could not tokenize prompt");
    }
    tokens.resize((size_t)actual_count);

    if (tokens.empty() || (int)tokens.size() >= self->context_size) {
        throw std::runtime_error("Prompt is empty or too long for this context");
    }

    llama_batch batch = llama_batch_get_one(
        tokens.data(), (int32_t)tokens.size());
    if (llama_decode(context, batch) != 0) {
        throw std::runtime_error("Could not process prompt");
    }

    llama_sampler *sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params());
    if (sampler == nullptr) {
        throw std::runtime_error("Could not create sampler");
    }

    if (temperature <= 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(
            sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    const int room = self->context_size - (int)tokens.size();
    const int output_limit = std::min(max_tokens, room);
    std::string answer;

    for (int i = 0; i < output_limit; ++i) {
        const llama_token token = llama_sampler_sample(sampler, context, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }
        llama_sampler_accept(sampler, token);

        char piece[256];
        const int piece_size = llama_token_to_piece(
            vocab, token, piece, sizeof(piece), 0, true);
        if (piece_size < 0) {
            llama_sampler_free(sampler);
            throw std::runtime_error("Could not convert output token to text");
        }
        answer.append(piece, (size_t)piece_size);

        llama_batch next_batch = llama_batch_get_one(&token, 1);
        if (llama_decode(context, next_batch) != 0) {
            llama_sampler_free(sampler);
            throw std::runtime_error("Could not generate next token");
        }
    }

    llama_sampler_free(sampler);
    return answer;
}

static PyObject *WBModel_chat(WBModel *self, PyObject *args, PyObject *kwargs)
{
    PyObject *messages = nullptr;
    int max_tokens = 256;
    float temperature = 0.7f;
    static char *names[] = {
        (char *)"messages",
        (char *)"max_tokens",
        (char *)"temperature",
        nullptr
    };

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O|if", names,
            &messages, &max_tokens, &temperature)) {
        return nullptr;
    }
    if (max_tokens < 1 || temperature < 0.0f) {
        PyErr_SetString(PyExc_ValueError,
                        "max_tokens must be positive and temperature non-negative");
        return nullptr;
    }

    try {
        const std::string prompt = make_prompt(messages);
        std::string answer;
        std::string error;

        PyThreadState *thread_state = PyEval_SaveThread();
        try {
            answer = generate_reply(self, prompt, max_tokens, temperature);
        } catch (const std::exception &exception) {
            error = exception.what();
        }
        PyEval_RestoreThread(thread_state);

        if (!error.empty()) {
            PyErr_SetString(PyExc_RuntimeError, error.c_str());
            return nullptr;
        }
        return PyUnicode_FromStringAndSize(answer.data(), (Py_ssize_t)answer.size());
    } catch (const std::exception &exception) {
        PyErr_SetString(PyExc_ValueError, exception.what());
        return nullptr;
    }
}

static PyMethodDef WBModel_methods[] = {
    {"chat", (PyCFunction)WBModel_chat, METH_VARARGS | METH_KEYWORDS,
     "Reply to a list of (role, text) messages."},
    {nullptr, nullptr, 0, nullptr}
};

static PyTypeObject WBModelType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
};

static PyModuleDef wb_module = {
    PyModuleDef_HEAD_INIT,
    "_wb_ai",
    "Local GGUF inference for WB-AI.",
    -1,
    nullptr
};

PyMODINIT_FUNC PyInit__wb_ai(void)
{
    llama_backend_init();

    WBModelType.tp_name = "wb_ai.Model";
    WBModelType.tp_basicsize = sizeof(WBModel);
    WBModelType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    WBModelType.tp_new = WBModel_new;
    WBModelType.tp_init = (initproc)WBModel_init;
    WBModelType.tp_dealloc = (destructor)WBModel_dealloc;
    WBModelType.tp_methods = WBModel_methods;
    WBModelType.tp_doc = "Load a local GGUF model and generate replies.";

    if (PyType_Ready(&WBModelType) < 0) {
        return nullptr;
    }

    PyObject *module = PyModule_Create(&wb_module);
    if (module == nullptr) {
        return nullptr;
    }

    Py_INCREF(&WBModelType);
    if (PyModule_AddObject(module, "Model", (PyObject *)&WBModelType) < 0) {
        Py_DECREF(&WBModelType);
        Py_DECREF(module);
        return nullptr;
    }
    return module;
}
