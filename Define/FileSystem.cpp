#include "WB.h"
#include <iostream>
#include <fstream>
#include <map>
#include <string>

struct WordEntry {
    std::string definition;
    std::string partOfSpeech;
    std::string sentence;
};

void Mini_AI::fileSaveKnowledge(const std::string& filename, const std::map<std::string, WordEntry>& dictionary) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    for (const auto& entry : dictionary) {
        outFile << entry.first << "|" 
                << entry.second.definition << "||" 
                << entry.second.partOfSpeech << "||" 
                << entry.second.sentence << "\n";
    }
    outFile.close();
}

void Mini_AI::FileLoadKnowledge(const std::string& filename, std::map<std::string, WordEntry>& dictionary) {
    std::ifstream inFile(filename);
    if (!inFile) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return;
    }
    std::string line;
    while (std::getline(inFile, line)) {
        size_t delimiterPos = line.find("|");
        if (delimiterPos != std::string::npos) {
            std::string word = line.substr(0, delimiterPos);
            std::string rest = line.substr(delimiterPos + 1);

            size_t first = rest.find("||");
            size_t second = rest.find("||", first + 2);

            std::string definition = rest.substr(0, first);
            std::string pos = rest.substr(first + 2, second - (first + 2));
            std::string sentence = rest.substr(second + 2);

            dictionary[word] = {definition, pos, sentence};
        }
    }
    inFile.close();
}
