void Mini_AI::aiTrain(int epochs, double lr) {
    learningRate = lr;
    epochsRun = 0;


    dictionary.clear();
    fileLoadKnowledge("ExampleWords.txt", dictionary);


    numericInputs.clear();
    numericOutputs.clear();
    for (auto& [word, meaning] : dictionary) {
        std::vector<double> features;


        features.push_back(static_cast<double>(word.length()));
        double sumAscii = 0; int vowels = 0;
        for (char c : word) {
            sumAscii += static_cast<int>(c);
            if (std::string("aeiouAEIOU").find(c) != std::string::npos) vowels++;
        }
        features.push_back(sumAscii / word.length());
        features.push_back(static_cast<double>(vowels));

        numericInputs.push_back(features);

        numericOutputs.push_back(!meaning.empty() ? 1.0 : 0.0);
    }

    numExamples = numericInputs.size();
    sizeOfVocab = dictionary.size();

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
            std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                      << " | Avg Loss: " << totalLoss / numericInputs.size()
                      << " | Vocab: " << sizeOfVocab
                      << " | Examples: " << numExamples
                      << std::endl;
        }
    }

    is_trained = true;
    std::ofstream out("trained_model.txt");
    for (auto& w : weights) out << w << " ";
    out << "\n" << bias << std::endl;
    out.close();
}
