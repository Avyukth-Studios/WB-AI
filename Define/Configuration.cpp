#include "WB.h"
//Setters
void Mini_AI::setThreshold(double threshold)
{
    this->threshold = threshold;
}

void Mini_AI::setFallBack(const std::string& fallBack)
{
  this->fallBack = fallBack;
}

void Mini_AI::setVerbose(bool verbose)
{
  this->verbose = verbose;
}

void Mini_AI::setSeed(int seed)
{
  this->seed = seed;
}
