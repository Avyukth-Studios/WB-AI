#include "WB.h"
#include <iostream>
#include <vector>
// Idk math for this. asked AI how to do these. Gave me the formulae;
double Mini_AI::math_forward(const std::vector<double>& input) 
{
  int TempSum = 0;
  for(auto& i : inputs)
  {
    sum += inputs[i] * weight[i];
  }
  sum += bias;

  prediction = 1 / (1 + exp(-sum))
  return prediction;
}

double Mini_AI::math_computeLoss(double prediction, double expected)
{
  diffrence = prediction - expected;
  double loss = diff * diff;
  return loss;
}

std::vector<double> Mini_AI::math_computeGradient(const std::vector<double>& input, double prediction, double expected)
{
    double whatever = 2 * (prediction - expected);
    double shitffff = prediction * (1-prediction);
    double result = whatever * shitffff;
    double gradient = input * result;
    double gradientBias = result;
    return gradientBias;
}

void Mini_AI::math_updateWeights(const std::vector<double>& gradient, double lr)
{
  weight = weight - (lr * gradient);
  bias = biasGradient - (lr * bias)
}

  
