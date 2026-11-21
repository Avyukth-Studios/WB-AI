#include "WB.h"
#include <iostream>

int main() {
    Mini_AI ai;
    ai.setVerbose(true);
    ai.addKnowledge("cpp=C++ is a general-purpose programming language.");
    ai.train_ai(20, 0.01);

    std::cout << ai.aiReply("What is cpp?") << '\n';
    std::cout << "Confidence: " << ai.aiLastConfidence() << '\n';
}
