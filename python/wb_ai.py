import ctypes
from pathlib import Path


class WB_AI:
    def __init__(self, library_path=None):
        if library_path is None:
            library_path = str(next(Path('build').glob('*wbai*.so'), 'build/libwbai.so'))

        self.lib = ctypes.CDLL(library_path)
        self.lib.ai_create.restype = ctypes.c_void_p
        self.lib.ai_destroy.argtypes = [ctypes.c_void_p]
        self.lib.ai_train.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        self.lib.ai_reply.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.ai_reply.restype = ctypes.c_char_p
        self.lib.ai_last_confidence.argtypes = [ctypes.c_void_p]
        self.lib.ai_last_confidence.restype = ctypes.c_double
        self.lib.ai_is_trained.argtypes = [ctypes.c_void_p]
        self.lib.ai_is_trained.restype = ctypes.c_int
        self.lib.ai_add_knowledge.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.ai_add_knowledge.restype = ctypes.c_int
        self.lib.ai_crawl_website.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.ai_crawl_website.restype = ctypes.c_int

        self.ai = self.lib.ai_create()
        if not self.ai:
            raise RuntimeError('Could not create WB-AI instance')

    def add_knowledge(self, text):
        return bool(self.lib.ai_add_knowledge(self.ai, text.encode('utf-8')))

    def train_ai(self, epochs=10, learning_rate=0.01):
        self.lib.ai_train(self.ai, epochs, learning_rate)

    def reply(self, text):
        result = self.lib.ai_reply(self.ai, text.encode('utf-8'))
        return result.decode('utf-8') if result else ''

    def crawl_website(self, url):
        return bool(self.lib.ai_crawl_website(self.ai, url.encode('utf-8')))

    def close(self):
        if self.ai:
            self.lib.ai_destroy(self.ai)
            self.ai = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
