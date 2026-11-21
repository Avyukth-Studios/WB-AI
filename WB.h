#ifndef AI_CORE_H
#define AI_CORE_H

#include <map>
#include <string>
#include <vector>

struct WordEntry {
    std::string definition;
    std::string partOfSpeech;
    std::string sentence;
};

class Mini_AI {
public:
    Mini_AI();

    void train_ai(int epochs = 10, double learningRate = 0.01);
    void aiTrain(int epochs, double learningRate);

    bool addKnowledge(const std::string& text);
    bool addKnowledgeFile(const std::string& filename);
    bool crawlWebsite(const std::string& url, const std::string& outputFile = "");

    std::string aiReply(const std::string& input);
    void aiReply(const std::vector<double>& input);
    void aiReply(const int& input);
    void aiReplyString(const std::string& input);
    double aiLastConfidence();

    void fileSaveKnowledge(const std::string& filename,
                           const std::map<std::string, WordEntry>& dictionary);
    void fileLoadKnowledge(const std::string& filename,
                           std::map<std::string, WordEntry>& dictionary);

    void setThreshold(double threshold);
    void setFallBack(const std::string& fallBack);
    void setVerbose(bool verbose);
    void setSeed(int seed);

    int numExamples();
    int numResponses();
    int vocabSize();
    bool isTrained();
    std::string lastError();

private:
    std::map<std::string, WordEntry> dictionary;
    std::vector<std::string> documents;
    std::vector<double> weights;
    std::vector<std::vector<double>> numericInputs;
    std::vector<double> numericOutputs;
    double bias;
    double learningRate;
    int epochsRun;
    bool is_trained;
    double last_Confidence;
    double threshold;
    std::string fallBack;
    int numExamplesCount;
    int numResponsesCount;
    int sizeOfVocab;
    bool verbose;
    int seed;
    std::string LastError;

    std::vector<double> features(const std::string& text) const;
    double similarity(const std::string& left, const std::string& right) const;
};

#endif
