#include "WB.h"  

bool Mini_AI::ai_is_trained() {
    return is_trained;
}

int Mini_AI::ai_num_examples()
{
    return numExamples;
}

int Mini_AI::ai_num_responses()
{
    return numResponses;
}

int Mini_AI::ai_vocab_size()
{
    return sizeOfVocab;
}

int Mini_AI::last_error()
{
    return lastErrorValue;
}
