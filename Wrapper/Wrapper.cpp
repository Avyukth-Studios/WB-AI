#include "Wrap.h"
#include "WB.h"
#include <cstdlib>
#include <cstring>

static thread_local std::string ai_reply_buffer;

void* ai_create() {
    try {
        return new Mini_AI();
    } catch (...) {
        return nullptr;
    }
}

void ai_destroy(void* ai) {
    delete static_cast<Mini_AI*>(ai);
}

void ai_train(void* ai, int epochs, double learningRate) {
    if (ai) static_cast<Mini_AI*>(ai)->train_ai(epochs, learningRate);
}

const char* ai_reply(void* ai, const char* input) {
    if (!ai || !input) return "";
    ai_reply_buffer = static_cast<Mini_AI*>(ai)->aiReply(input);
    return ai_reply_buffer.c_str();
}

const char* ai_last_error(void* ai) {
    if (!ai) return "";
    ai_reply_buffer = static_cast<Mini_AI*>(ai)->lastError();
    return ai_reply_buffer.c_str();
}

double ai_last_confidence(void* ai) {
    return ai ? static_cast<Mini_AI*>(ai)->aiLastConfidence() : 0.0;
}

int ai_is_trained(void* ai) {
    return ai ? static_cast<Mini_AI*>(ai)->isTrained() : 0;
}

int ai_add_knowledge(void* ai, const char* text) {
    return ai && text ? static_cast<Mini_AI*>(ai)->addKnowledge(text) : 0;
}

int ai_crawl_website(void* ai, const char* url) {
    return ai && url ? static_cast<Mini_AI*>(ai)->crawlWebsite(url) : 0;
}
