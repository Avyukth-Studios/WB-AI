from ._wb_ai import Model


class ChatBot:
    """Small conversation helper around the native local model."""

    def __init__(self, model_path, context_size=4096):
        self._model = Model(model_path, context_size)
        self._messages = []

    def ask(self, text, max_tokens=256, temperature=0.7):
        self._messages.append(("user", text))
        answer = self._model.chat(
            self._messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._messages.append(("assistant", answer))
        return answer

    def clear(self):
        self._messages.clear()

    @property
    def messages(self):
        return list(self._messages)
