#include "WB.h"
//Setters
void Mini_AI::ai_set_threshold(double& threshold)
{
    this->threshold = threshold;
}

void Mini_AI::ai_set_fallBack(const std::string& fallBack)
{
  this->fallBack = falBack;
}

void Mini_AI::ai_set_verbose(bool verbose)
{
  this->verbose = verbose;
}

void Mini_AI::ai_set_seed(int seed)
{
  this->seed = seed;
}
