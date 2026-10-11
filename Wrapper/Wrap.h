#ifndef AI_CORE_C_H
#define AI_CORE_C_H

#include "WB.h"

#ifdef __cplusplus
extern "C" {
#endif

// Lifecycle
void* ai_create();
void ai_destroy(void* ai);

// Training
void ai_train(void* ai, int epochs, double lr);

// Replies
void ai_replyString(void* ai, const char* input);
double ai_lastConfidence(void* ai);

#ifdef __cplusplus
}
#endif

#endif
