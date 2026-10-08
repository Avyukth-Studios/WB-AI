#ifndef AI_CORE_H
#define AI_CORE_H

#include <iostream>
#include <vector>
#include <string>

class Mini_AI
{
public: 
    //Initialize the Variables
    Mini_AI()
    {
        //AI state
        is_Trained = false; // Is the AI trained?
        last_Confidence = 0.0; // Confidence score from last reply
        threshold  = 0.0; // Minimum Confidence before fallback
        fallBack = "404 nothing to see here!"; // Default reply if confidence too low

        //Stats & metaDeta
        numExamples = 0; // count of training examples
        numResponses = 0; // Count of responses Stored
        sizeOfVocab = 0; // Basically Unique Words

        // Config & Debug
        verbose = false; //Toggle detail Logging
        seed = 0; // Random seed for reproducability
        LastError = "404 nothing to see here!"; // Log the last error
    }

    //training
    void trainLAlgebra(int epohs, double lr);

    //file system
    void SAVETOTXT(const std::string& filename, int AIstate); //AI state will be 0 or 1. 0 for off and 1 for on
    void LOADFROMTXT(const std::string& filename, int AIstate); //change the AI state.

    //LifeCycle
    void ai_create();
    void ai_destroy();

    //Data Management
    void ai_reply(const std::vector<double>& input);
    void ai_last_Confidence();
    void ai_accuracy(const std::vector<double>& expectedOutputs);

    //Configuration
    void ai_set_threshold(double threshold);
    void ai_set_fallBack(const std::string& fallBack);
    void ai_set_verbose(bool verbose);
    void ai_set_seed(int seed);

    //Statistics
    void ai_num_examples();
    void ai_num_responses();
    void ai_vocab_size();
    void ai_is_trained();
    void ai_last_error();

    //Helper Functions
    double forward(const std::vector<double>& input);
    double computeLoss(double prediction, double expected);
    std::vector<double> computeGradient(const std::vector<double>& input, double prediction, double expected);
    void updateWeights(const std::vector<double>& gradient, double lr);


private:
    //storing I/O
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<double> weights; // Adjustable parameters for input feature
    std::vector<std::vector<double>> numericInputs; //Each training example as a vector of numbers
    std::vector<double> numericOutputs; // Expected Numeric Results
    double bias; // Predctions

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

#endif AI_CORE_H
        
