#ifndef AI_CORE_H
#define AI_CORE_H

#include <iostream>
#include <vector>
#include <string>
#include <map>

struct WordEntry {
    std::string definition;
    std::string partOfSpeech;
    std::string sentence;
};

class Mini_AI
{
public:
    Mini_AI();

    // Training
    void aiTrain(int epochs, double lr);

    // File system
    void fileSaveKnowledge(const std::string& filename, const std::map<std::string, WordEntry>& dictionary);
    void fileLoadKnowledge(const std::string& filename, std::map<std::string, WordEntry>& dictionary);

    // Data management
    void aiReply(const std::vector<double>& input);
    void aiReply(const std::string& input);
    void aiReply(const int& input);

    // Backwards-compatible aliases for the old naming scheme.
    void aiReplyString(const std::string& input) { aiReply(input); }
    void aiReplyNumeric(const int& input) { aiReply(input); }

    double aiLastConfidence();
    void aiAccuracy(const std::vector<double>& expectedOutputs);

    // Configuration
    void setThreshold(double threshold);
    void setFallBack(const std::string& fallBack);
    void setVerbose(bool verbose);
    void setSeed(int seed);

    // Statistics
    int numExamples();
    int numResponses();
    int vocabSize();
    bool isTrained();
    std::string lastError();

    // Helper functions
    double math_forward(const std::vector<double>& input);
    double math_computeLoss(double prediction, double expected);
    std::vector<double> math_computeGradient(const std::vector<double>& input, double prediction, double expected);
    void math_updateWeights(const std::vector<double>& gradient, double lr);

private:
    // Storing I/O
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<double> weights;
    std::vector<std::vector<double>> numericInputs;
    std::vector<double> numericOutputs;
    double bias;
    std::map<std::string, WordEntry> dictionary;

    // AI training
    double learningRate;
    int epochsRun;

    // AI state
    bool is_trained;
    double last_Confidence;
    double threshold;
    std::string fallBack;

    // Stats & metadata
    int numExamples;
    int numResponses;
    int sizeOfVocab;

    // Config & debug
    bool verbose;
    int seed;
    std::string LastError;
    double lastErrorValue;
};

#endif
