#ifndef AI_CORE_H
#define AI_CORE_H

#include <iostream>
#include <vector>
#include <string>
#include <map>

class Mini_AI
{
public: 
    //Initialize the Variables
    Mini_AI()
    {
        //AI state
        is_trained = false; // Is the AI trained?
        last_Confidence = 0.0; // Confidence score from last reply
        threshold  = 0.0; // Minimum Confidence before fallback
        fallBack = "NULLSTRING"; // Default reply if confidence too low

        //Stats & metaDeta
        numExamples = 0; // count of training examples
        numResponses = 0; // Count of responses Stored
        sizeOfVocab = 0; // Basically Unique Words

        // Config & Debug
        verbose = false; //Toggle detail Logging
        seed = 0; // Random seed for reproducability
        LastError = "404 nothing to see here!"; // Log the last error

        std::cout << "WorkBench AI library." << std::endl;
        std::cout << "© 2025 Avyukth Studios. All Rights Reserved Unpublished";
    }
      
    //training (Pending)
    void aiTrain(int epohs, double lr);

    //file systemV (DONE)
    void fileSaveKnowledge(const std::string& filename, const std::map<std::string, std::string>& dictionary); //Save the knowledge
    void fileLoadKnowledge(const std::string& filename, std::map<std::string, std::string>& dictionary); //Load Knowledge


    //Data Management (PENDING 2 functions)
    void aiReply(const std::vector<double>& input);
    void aiLastConfidence();
    void aiAccuracy(const std::vector<double>& expectedOutputs);
    void aiReplyString(const std::string& input); //pending
    void aiReplyNumeric(const int& input); //pending

    //Configuration (DONE)
    void setThreshold(double& threshold);
    void setFallBack(const std::string& fallBack);
    void setVerbose(bool verbose);
    void setSeed(int seed);

    //Statistics (DONE)
    void numExamples();
    void numResponses();
    void vocabSize();
    bool isTrained();
    int lastError();

    //Helper Functions (DONE)
    double math_forward(const std::vector<double>& input);
    double math_computeLoss(double prediction, double expected);        
    std::vector<double> math_computeGradient(const std::vector<double>& input, double prediction, double expected);
    void math_updateWeights(const std::vector<double>& gradient, double lr);


private:
    //storing I/O
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<double> weights; // Adjustable parameters for input feature
    std::vector<std::vector<double>> numericInputs; // Each training example as a vector of numbers
    std::vector<double> numericOutputs; // Expected Numeric Results
    double bias; // Predctions
    std::map<std::string, std::string> dictionary; // Storing vocabulary

    //AI training
    double learningRate; // Updates
    int epochsRun; // How many Epochs Completed?

    //AI state
    bool is_Trained; // Is the AI trained?
    double last_Confidence; // Confidence score from last reply
    double threshold; // Minimum Confidence before fallback
    std::string fallBack; // Default reply if confidence too low

    //Stats & metaDeta
    int numExamples; // count of training examples
    int numResponses; // Count of responses Stored
    int sizeOfVocab; // Basically Unique Words

    // Config & Debug
    bool verbose; //Toggle detail Logging
    int seed; // Random seed for reproducability
    std::string LastError; // Log the last error
    double lastErrorValue; // most recent loss of value
};

#endif
