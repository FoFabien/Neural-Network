#include <iostream>
#include <cmath>

#include "neuralnetwork.hpp"

void loadAndRun(const std::string &file)
{
    NeuralNetwork test;
    if(test.load(file))
    {
        std::cout << "Network loaded from " << file << std::endl;
        for(size_t i = 0; i < 4; ++i)
        {
            test.readyNetwork();
            std::vector<double> inputs;
            switch(i)
            {
                case 1: inputs = {0, 1}; std::cout << "Input(s): 0 1 "; break;
                case 2: inputs = {1, 0}; std::cout << "Input(s): 1 0 "; break;
                case 3: inputs = {1, 1}; std::cout << "Input(s): 1 1 "; break;
                default: inputs = {0, 0}; std::cout << "Input(s): 0 0 "; break;
            }
            test.setInput(inputs);

            const std::vector<double>& res = test.getOutputs();
            std::cout << " | Output(s): ";
            for(const double& r : res)
                if(r > 0.7) std::cout << "1 (" << r << ")";
                else if(r < 0.3) std::cout << "0 (" << r << ")";
                else std::cout << "? (" << r << ") ";
            std::cout << std::endl;
        }
    }
    else
        std::cout << "Can't load the neural network from " << file << std::endl;
}

bool train(NeuralNetwork &test)
{
    std::vector<std::vector<double> > inputs = {{0, 0}, {0, 1}, {1, 1}, {1, 0}};
    std::vector<std::vector<double> > outputs = {{0}, {1}, {0}, {1}};
    return test.runTraining(inputs, outputs, 100000, inputs.size(), 0.8, 0.2, 0.001, false, "");
}

void createAndTrain(const std::string &file)
{
    NeuralNetwork test({2, 2, 1});
    test.autoConnect();
    test.initNetwork();
    if(train(test) && test.save(file))
        std::cout << "Network saved to " << file << std::endl;
}

void resumeTraining(const std::string &file)
{
    NeuralNetwork test;
    if(test.load(file))
    {
        std::cout << "Network loaded from " << file << std::endl;
        if(train(test) && test.save(file))
            std::cout << "Network saved to " << file << std::endl;
    }
    else
        std::cout << "Can't load the neural network from " << file << std::endl;
}

int main(int argc, char *argv[])
{
    std::cout << "Hello world!" << std::endl;
    size_t type = 0;
    for(int i = 1; i < argc-1; ++i)
    {
        if(std::string(argv[i]) == "-n")
            type = 1;
        else if(std::string(argv[i]) == "-r")
            type = 2;
        else if(std::string(argv[i]) == "-l")
            type = 3;
    }
    std::string file = argv[argc-1];

    if(argc >= 2 && type > 0 && !file.empty())
    {
        switch(type)
        {
            case 1: createAndTrain(file); break;
            case 2: resumeTraining(file); break;
            case 3: loadAndRun(file); break;
        }
    }
    else
    {
        std::cout << "usage: app.exe [-n|-r|-l] filename" << std::endl;
        std::cout << "\t-n : create a new network, train and save it" << std::endl;
        std::cout << "\t-r : load and resume the network training" << std::endl;
        std::cout << "\t-l : load the network and run it" << std::endl;
    }

    return 0;
}
