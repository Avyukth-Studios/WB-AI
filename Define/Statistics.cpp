#include "WB.h"  

bool Mini_AI::isTrained() {
    return is_trained;
}

int Mini_AI::numExamples()
{
    return numExamples;
}

int Mini_AI::numResponses()
{
    return numResponses;
}

int Mini_AI::vocabSize()
{
    return sizeOfVocab;
}

double Mini_AI::aiLastConfidence()
{
    return last_Confidence;
}

int Mini_AI::lastError()
{
    return static_cast<int>(lastErrorValue);
}
