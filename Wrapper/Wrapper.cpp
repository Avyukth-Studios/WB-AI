#include "Wrap.h"
#include "WB.h"  

void* ai_create() {
    return new Mini_AI();
}

void ai_destroy(void* ai) {
    delete static_cast<Mini_AI*>(ai);
}

void ai_train(void* ai, int epochs, double lr) {
    static_cast<Mini_AI*>(ai)->aiTrain(epochs, lr);
}

void ai_replyString(void* ai, const char* input) {
    static_cast<Mini_AI*>(ai)->aiReplyString(input);
}

double ai_lastConfidence(void* ai) {
    return static_cast<Mini_AI*>(ai)->aiLastConfidence();
}
