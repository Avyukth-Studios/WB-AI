#include<iostream>
#include<vector>
#define AI_CORE_H
#endif AI_CORE_H

struct Mini_AI
{
    //storing I/O
    std::vector<std::string> inputs;
    std::vector<std::string> output;

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
};
class Mini_AI_Behaviour
{
    public: 

    //Initialize the Variables
    Mini_AI()
    {
        Mini_AI MA
        MA.is_Trained = false;
        MA.last_Confidence = 0.0;
        MA.threshold  = 0.0;
        MA.fallBack = "404 nothing to see here!";
        MA.numExamples = 0;
        MA.numResponses = 0;
        MA.sizeOfVocab = 0;
        MA.verbose = false;
        MA.seed = 0;
        MA.LastError = "404 nothing to see here!";
    }

    //training
        void trainLAlgebra(int epohs, double lr);

    //file system
        void SAVETOTXT(std::string& filename, int AIstate); //AI state will be 0 or 1. 0 for off and 1 for on
        void LOADFROMTXT(std::string& filename, int AIstate); //change the AI state.

    //LifeCycle
        void ai_create();
        void ai_destroy();

    //Data Management
        void ai_reply(const std::string& input);
        void ai_last_Confidence();
        void ai_accuracy();

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
};


