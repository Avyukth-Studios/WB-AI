#ifndef AI_CORE_C_H
#define AI_CORE_C_H

#ifdef __cplusplus
extern "C" {
#endif

void* ai_create();
void ai_destroy(void* ai);
void ai_train(void* ai, int epochs, double learningRate);
const char* ai_reply(void* ai, const char* input);
const char* ai_last_error(void* ai);
double ai_last_confidence(void* ai);
int ai_is_trained(void* ai);
int ai_add_knowledge(void* ai, const char* text);
int ai_crawl_website(void* ai, const char* url);

#ifdef __cplusplus
}
#endif

#endif
