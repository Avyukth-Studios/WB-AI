#include "WB.h"
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <cmath>

double Mini_AI::math_forward(const std::vector<double>& input) {
    double sum = bias;
    for (size_t i = 0; i < input.size(); ++i) {
        sum += input[i] * weights[i];
    }
    return sum;
}

double Mini_AI::math_computeLoss(double prediction, double expected) {
    double diff = prediction - expected;
    lastErrorValue = 0.5 * diff * diff;
    return lastErrorValue;
}

std::vector<double> Mini_AI::math_computeGradient(const std::vector<double>& input, double prediction, double expected) {
    double diff = prediction - expected;
    std::vector<double> grad(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        grad[i] = diff * input[i];
    }
    return grad;
}

void Mini_AI::math_updateWeights(const std::vector<double>& gradient, double lr) {
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= lr * gradient[i];
    }
    bias -= lr * (gradient.empty() ? 0.0 : gradient[0]); 
}

void Mini_AI::aiTrain(int epochs, double lr) {
    learningRate = lr;
    epochsRun = 0;

    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<double> dist(-0.01, 0.01);
    weights.resize(numericInputs[0].size());
    for (auto& w : weights) w = dist(gen);
    bias = dist(gen);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;
        for (size_t i = 0; i < numericInputs.size(); ++i) {
            double prediction = math_forward(numericInputs[i]);
            double loss = math_computeLoss(prediction, numericOutputs[i]);
            totalLoss += loss;
            std::vector<double> grad = math_computeGradient(numericInputs[i], prediction, numericOutputs[i]);
            math_updateWeights(grad, learningRate);
        }
        epochsRun++;
        if (verbose) {
            std::cout << "Epoch " << epoch+1 << "/" << epochs 
                      << " Loss: " << totalLoss / numericInputs.size() << std::endl;
        }
    }

    is_trained = true;
    std::ofstream out("trained_model.txt");
    for (auto& w : weights) out << w << " ";
    out << "\n" << bias << std::endl;
    out.close();
}

void Mini_AI::fileSaveKnowledge(const std::string& filename, const std::map<std::string, std::string>& dict) {
    std::ofstream out(filename);
    for (auto& [key, val] : dict) {
        out << key << ":" << val << "\n";
    }
    out.close();
}

void Mini_AI::fileLoadKnowledge(const std::string& filename, std::map<std::string, std::string>& dict) {
    std::ifstream in(filename);
    std::string line;
    while (std::getline(in, line)) {
        size_t pos = line.find(':');
        if (pos != std::string::npos) {
            dict[line.substr(0,pos)] = line.substr(pos+1);
        }
    }
    in.close();
}

void Mini_AI::aiReply(const std::vector<double>& input) {
    double prediction = math_forward(input);
    last_Confidence = prediction;
    std::cout << "Reply confidence: " << prediction << std::endl;
}

void Mini_AI::aiLastConfidence() {
    std::cout << "Last confidence: " << last_Confidence << std::endl;
}

void Mini_AI::aiAccuracy(const std::vector<double>& expectedOutputs) {
    double correct = 0;
    for (size_t i = 0; i < expectedOutputs.size(); ++i) {
        double pred = math_forward(numericInputs[i]);
        if ((pred >= 0.5 && expectedOutputs[i] == 1.0) ||
            (pred < 0.5 && expectedOutputs[i] == 0.0)) {
            correct++;
        }
    }
    std::cout << "Accuracy: " << (correct / expectedOutputs.size()) * 100.0 << "%" << std::endl;
}

void Mini_AI::setThreshold(double& t) { threshold = t; }
void Mini_AI::setFallBack(const std::string& f) { fallBack = f; }
void Mini_AI::setVerbose(bool v) { verbose = v; }
void Mini_AI::setSeed(int s) { seed = s; }

void Mini_AI::numExamples() { std::cout << "Examples: " << numExamples << std::endl; }
void Mini_AI::numResponses() { std::cout << "Responses: " << numResponses << std::endl; }
void Mini_AI::vocabSize() { std::cout << "Vocab size: " << sizeOfVocab << std::endl; }
bool Mini_AI::isTrained() { return is_trained; }
int Mini_AI::lastError() { std::cout << LastError << std::endl; return 0; }
