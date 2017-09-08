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
            switch(i)
            {
                case 1: test.setInput({0, 1}); std::cout << "Input(s): 0 1 "; break;
                case 2: test.setInput({1, 0}); std::cout << "Input(s): 1 0 "; break;
                case 3: test.setInput({1, 1}); std::cout << "Input(s): 1 1 "; break;
                default: test.setInput({0, 0}); std::cout << "Input(s): 0 0 "; break;
            }

            std::vector<float> res = test.getOutputs();
            std::cout << " | Output(s): ";
            for(float& r : res)
                if(r > 0.9) std::cout << "1 ";
                else if(r < 0.1) std::cout << "0 ";
                else std::cout << "error ";
            std::cout << std::endl;
        }
    }
    else
        std::cout << "Can't load the neural network from " << file << std::endl;
}


void train(NeuralNetwork &test)
{
    std::vector<std::vector<float> > inputs = {{0, 0}, {0, 1}, {1, 1}, {1, 0}};
    std::vector<std::vector<float> > outputs = {{0}, {1}, {0}, {1}};
    test.runTraining(inputs, outputs, 100000, 0.8, true);
}

void createAndTrain(const std::string &file)
{
    NeuralNetwork test({2, 2, 1});
    test.autoConnect();
    test.initNetwork();
    train(test);
    if(test.save(file))
        std::cout << "Network saved to " << file << std::endl;
}

void resumeTraining(const std::string &file)
{
    NeuralNetwork test;
    if(test.load(file))
    {
        std::cout << "Network loaded from " << file << std::endl;
        train(test);
        if(test.save(file))
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
