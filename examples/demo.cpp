#include "WB.h"
#include <iostream>
#include <string>

int main() {
    Mini_AI ai;
    if (!ai.addKnowledgeFile("knowledge.txt")) {
        ai.addKnowledge("cpp=C++ is a general-purpose programming language.");
        ai.addKnowledge("wb-ai=WB-AI is a C++ knowledge retrieval library.");
    }
    ai.train_ai(20, 0.01);

    std::cout << "WB-AI is ready. Ask a question, or type /quit to exit.\n";
    std::string question;
    while (std::cout << "> " && std::getline(std::cin, question)) {
        if (question == "/quit") break;
        if (question.empty()) continue;
        std::cout << ai.aiReply(question) << '\n';
    }
}
