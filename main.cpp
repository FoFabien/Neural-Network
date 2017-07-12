#include <iostream>
#include <utility>
#include <chrono>
#include <thread>

#include "neuralnetwork.hpp"

void loadAndRun()
{
    NeuralNetwork test;
    if(test.load("network.txt"))
    {
        std::cout << "Network loaded" << std::endl;
        test.readyNetwork();
        test.setInput({0, 1});
        std::vector<float> res = test.getOutputs();
        std::cout << "Input(s): 0 1 ";
        std::cout << " | Output(s): ";
        for(float& r : res)
            std::cout << r << " ";
        std::cout << std::endl;
    }
    else
        std::cout << "Can't load the neural network" << std::endl;
}

void createAndTrain()
{
    std::vector<std::vector<float> > inputs = {{0, 0}, {0, 1}, {1, 1}, {1, 0}};
    std::vector<std::vector<float> > outputs = {{0}, {1}, {0}, {1}};
    size_t hlayer_size = 2;
    NeuralNetwork test({2, hlayer_size, 1});
    for(size_t j = 0; j < hlayer_size; j++) // number of neurons on layer 2
    {
        for(size_t i = 0; i < 2; i++) // number of neurons on layer 1
        {
            test.connectNeurons(NeuCoor(0, i), NeuCoor(1, j)); // connecting 1st to 2nd layer
        }
        test.connectNeurons(NeuCoor(1, j), NeuCoor(2, 0)); // connecting 2nd to 3rd layer
    }
    test.initNetwork();
    test.print();
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    size_t max_step = 2000;
    for(size_t i = 0; i < max_step; i++)
    {
        if((i+1) % 1000 == 0)
        {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "### step: " << i+1 << " (" << (100*(i+1)/(max_step*1.f)) << "%) : elapsed = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
        }

        if(!test.train(inputs, outputs, 0.05, 1000, (i == max_step-1)))
        {
            std::cout << "error at step " << i << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    test.print();
    if(test.save("network.txt"))
        std::cout << "Network saved" << std::endl;
}

int main(int argc, char *argv[])
{
    std::cout << "Hello world!" << std::endl;
    if(argc >= 2 && std::string(argv[1]) == "-n")
        createAndTrain();
    else
        loadAndRun();

    return 0;
}
