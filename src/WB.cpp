#include "WB.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <curl/curl.h>

Mini_AI::Mini_AI()
    : bias(0.0), learningRate(0.01), epochsRun(0), is_trained(false),
      last_Confidence(0.0), threshold(0.35), fallBack("I do not have enough information to answer that."),
      numExamplesCount(0), numResponsesCount(0), sizeOfVocab(0), verbose(false), seed(42),
      LastError("") {}

static std::string normalize(const std::string& text) {
    std::string result;
    for (unsigned char c : text) {
        if (std::isalnum(c)) result += static_cast<char>(std::tolower(c));
        else if (!result.empty() && result.back() != ' ') result += ' ';
    }
    return result;
}

static std::vector<std::string> words(const std::string& text) {
    std::istringstream stream(normalize(text));
    std::vector<std::string> result;
    std::string word;
    while (stream >> word) result.push_back(word);
    return result;
}

static size_t receiveResponse(char* data, size_t size, size_t count, void* context) {
    const size_t bytes = size * count;
    std::string* response = static_cast<std::string*>(context);
    const size_t limit = 5 * 1024 * 1024;
    if (bytes > limit - response->size()) return 0;
    response->append(data, bytes);
    return bytes;
}

std::vector<double> Mini_AI::features(const std::string& text) const {
    auto tokens = words(text);
    if (tokens.empty()) return {0.0, 0.0, 0.0};

    double length = 0.0;
    double ascii = 0.0;
    double vowels = 0.0;
    for (const auto& token : tokens) {
        length += token.size();
        for (char c : token) {
            ascii += static_cast<unsigned char>(c);
            if (std::string("aeiou").find(c) != std::string::npos) ++vowels;
        }
    }
    return {length / tokens.size(), ascii / std::max<size_t>(1, length), vowels / std::max<size_t>(1, length)};
}

double Mini_AI::similarity(const std::string& left, const std::string& right) const {
    auto a = words(left);
    auto b = words(right);
    std::set<std::string> sa(a.begin(), a.end());
    std::set<std::string> sb(b.begin(), b.end());
    if (sa.empty() || sb.empty()) return 0.0;

    int intersection = 0;
    for (const auto& word : sa) intersection += sb.count(word);
    int unionSize = static_cast<int>(sa.size() + sb.size() - intersection);
    return unionSize ? static_cast<double>(intersection) / unionSize : 0.0;
}

bool Mini_AI::addKnowledge(const std::string& text) {
    if (text.empty()) {
        LastError = "Knowledge text is empty";
        return false;
    }
    documents.push_back(text);
    numExamplesCount = static_cast<int>(documents.size());
    numResponsesCount = static_cast<int>(documents.size());

    std::stringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        auto found = line.find('=');
        if (found != std::string::npos) {
            std::string key = line.substr(0, found);
            std::string rest = line.substr(found + 1);
            dictionary[key] = {rest, "", line};
        }
    }
    sizeOfVocab = static_cast<int>(dictionary.size());
    return true;
}

bool Mini_AI::addKnowledgeFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        LastError = "Unable to open knowledge file: " + filename;
        return false;
    }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return addKnowledge(text);
}

bool Mini_AI::crawlWebsite(const std::string& url, const std::string& outputFile) {
    if (url.empty()) {
        LastError = "Website URL is empty";
        return false;
    }

    static const bool curlReady = curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK;
    if (!curlReady) {
        LastError = "Unable to initialize the HTTP client";
        return false;
    }

    CURL* handle = curl_easy_init();
    if (!handle) {
        LastError = "Unable to create the HTTP request";
        return false;
    }

    std::string html;
    curl_easy_setopt(handle, CURLOPT_URL, url.c_str());
    curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, receiveResponse);
    curl_easy_setopt(handle, CURLOPT_WRITEDATA, &html);
    curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(handle, CURLOPT_MAXREDIRS, 5L);
    curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(handle, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(handle, CURLOPT_USERAGENT, "WB-AI/1.0");
    curl_easy_setopt(handle, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(handle, CURLOPT_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS);
    curl_easy_setopt(handle, CURLOPT_REDIR_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS);

    CURLcode result = curl_easy_perform(handle);
    curl_easy_cleanup(handle);
    if (result != CURLE_OK) {
        LastError = std::string("Website extraction failed: ") + curl_easy_strerror(result);
        return false;
    }

    std::string text;
    bool insideTag = false;
    for (char c : html) {
        if (c == '<') insideTag = true;
        else if (c == '>') {
            insideTag = false;
            text += ' ';
        } else if (!insideTag) {
            text += c;
        }
    }
    std::replace(text.begin(), text.end(), '\n', ' ');
    if (text.empty()) {
        LastError = "The website did not contain readable text";
        return false;
    }

    if (!outputFile.empty()) {
        std::ofstream out(outputFile);
        if (!out) {
            LastError = "Unable to write extracted website text: " + outputFile;
            return false;
        }
        out << text;
    }
    return addKnowledge(text);
}

void Mini_AI::train_ai(int epochs, double lr) {
    if (epochs <= 0 || !std::isfinite(lr) || lr <= 0.0) {
        LastError = "Training requires positive epochs and learning rate";
        return;
    }
    learningRate = lr;
    epochsRun = 0;
    numericInputs.clear();
    numericOutputs.clear();

    if (documents.empty()) {
        LastError = "No training data available";
        return;
    }

    std::mt19937 generator(seed == 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<double> distribution(-0.01, 0.01);

    weights.clear();
    for (int i = 0; i < 3; ++i) weights.push_back(distribution(generator));
    bias = distribution(generator);

    for (const auto& document : documents) {
        numericInputs.push_back(features(document));
        numericOutputs.push_back(1.0);
    }

    numExamplesCount = static_cast<int>(numericInputs.size());
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < numericInputs.size(); ++i) {
            double prediction = 0.0;
            for (size_t j = 0; j < weights.size(); ++j) prediction += weights[j] * numericInputs[i][j];
            prediction = 1.0 / (1.0 + std::exp(-(prediction + bias)));

            double error = numericOutputs[i] - prediction;
            for (size_t j = 0; j < weights.size(); ++j)
                weights[j] += learningRate * error * prediction * (1.0 - prediction) * numericInputs[i][j];
            bias += learningRate * error * prediction * (1.0 - prediction);
        }
        ++epochsRun;
    }

    is_trained = true;
    LastError.clear();
}

void Mini_AI::aiTrain(int epochs, double lr) { train_ai(epochs, lr); }

std::string Mini_AI::aiReply(const std::string& input) {
    if (input.empty()) {
        last_Confidence = 0.0;
        LastError = "Empty input";
        return fallBack;
    }

    double best = -1.0;
    std::string answer;
    for (const auto& [key, entry] : dictionary) {
        double score = similarity(input, key);
        if (entry.definition.empty()) continue;
        if (score > best) {
            best = score;
            answer = entry.definition;
        }
    }

    if (answer.empty()) {
        for (const auto& document : documents) {
            double score = similarity(input, document);
            if (score > best) {
                best = score;
                std::string documentText = document;
                if (documentText.size() > 500) documentText.resize(500);
                answer = documentText;
            }
        }
    }

    last_Confidence = best < 0.0 ? 0.0 : best;
    ++numResponsesCount;
    if (last_Confidence < threshold || answer.empty()) {
        LastError = "Confidence below threshold";
        return fallBack;
    }
    LastError.clear();
    return answer;
}

void Mini_AI::aiReply(const std::vector<double>&) {}
void Mini_AI::aiReply(const int&) {}
void Mini_AI::aiReplyString(const std::string& input) { aiReply(input); }
double Mini_AI::aiLastConfidence() { return last_Confidence; }

void Mini_AI::fileSaveKnowledge(const std::string& filename,
                                const std::map<std::string, WordEntry>& knowledge) {
    std::ofstream file(filename, std::ios::trunc);
    if (!file) {
        LastError = "Unable to save knowledge file: " + filename;
        return;
    }
    for (const auto& [key, entry] : knowledge)
        file << key << '=' << entry.definition << "||" << entry.partOfSpeech << "||" << entry.sentence << '\n';
}

void Mini_AI::fileLoadKnowledge(const std::string& filename,
                                std::map<std::string, WordEntry>& knowledge) {
    knowledge.clear();
    std::ifstream file(filename);
    if (!file) {
        LastError = "Unable to open knowledge file: " + filename;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        auto firstEqual = line.find('=');
        if (firstEqual == std::string::npos) continue;
        std::string key = line.substr(0, firstEqual);
        std::string rest = line.substr(firstEqual + 1);
        auto firstSeparator = rest.find("||");
        auto secondSeparator = firstSeparator == std::string::npos ? std::string::npos : rest.find("||", firstSeparator + 2);
        if (firstSeparator == std::string::npos || secondSeparator == std::string::npos)
            knowledge[key] = {rest, "", ""};
        else
            knowledge[key] = {rest.substr(0, firstSeparator),
                              rest.substr(firstSeparator + 2, secondSeparator - firstSeparator - 2),
                              rest.substr(secondSeparator + 2)};
    }
}

void Mini_AI::setThreshold(double value) { threshold = value; }
void Mini_AI::setFallBack(const std::string& value) { fallBack = value; }
void Mini_AI::setVerbose(bool value) { verbose = value; }
void Mini_AI::setSeed(int value) { seed = value; }
int Mini_AI::numExamples() { return numExamplesCount; }
int Mini_AI::numResponses() { return numResponsesCount; }
int Mini_AI::vocabSize() { return sizeOfVocab; }
bool Mini_AI::isTrained() { return is_trained; }
std::string Mini_AI::lastError() { return LastError; }
