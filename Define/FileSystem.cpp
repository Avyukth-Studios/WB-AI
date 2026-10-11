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

void Mini_AI::fileLoadKnowledge(const std::string& filename, std::map<std::string, WordEntry>& dictionary) {
    std::ifstream inFile(filename);
    if (!inFile) {
        std::cerr << "(© 2025 Avyukth Studios. All rights reserved unpublished)Error opening file for reading: " << filename << std::endl;
        return;
    }
    std::string line;
    int count = 0; // counter
    while (std::getline(inFile, line)) {
        size_t eqPos = line.find("=");
        if (eqPos == std::string::npos) continue;

        std::string word = line.substr(0, eqPos);
        std::string rest = line.substr(eqPos + 1);

        size_t first = rest.find("||");
        size_t second = rest.find("||", first + 2);
        if (first == std::string::npos || second == std::string::npos) continue;

        std::string definition = rest.substr(0, first);
        std::string pos = rest.substr(first + 2, second - (first + 2));
        std::string sentence = rest.substr(second + 2);

        dictionary[word] = {definition, pos, sentence};
        ++count; // increment
    }
    inFile.close();
    std::cout << "(© 2025 Avyukth Studios. All rights reserved unpublished)Loaded " << count << " entries from " << filename << std::endl;
}


void Mini_AI::fileSaveKnowledge(const std::string& filename, const std::map<std::string, WordEntry>& dictionary) {
    std::ofstream outFile(filename, std::ios::out | std::ios::trunc);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    for (const auto& [word, entry] : dictionary) {
        outFile << word << "="
                << entry.definition << "||"
                << entry.partOfSpeech << "||"
                << entry.sentence << "\n";
    }

    outFile.close();
}
