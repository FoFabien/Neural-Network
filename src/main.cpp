#include <iostream>
#include <utility>
#include <chrono>
#include <thread>

#include "neuralnetwork.hpp"

int main()
{
    std::cout << "Hello world!" << std::endl;

    std::vector<std::vector<float> > inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<float> > outputs = {{0}, {1}, {1}, {0}};
    NeuralNetwork test({2, 3, 1});
    for(size_t j = 0; j < 3; j++)
    {
        for(size_t i = 0; i < 2; i++)
        {
            test.connectNeurons(NeuCoor(0, i), NeuCoor(1, j));
        }
        test.connectNeurons(NeuCoor(1, j), NeuCoor(2, 0));
    }
    test.initNetwork();
    for(size_t i = 0; i < 300; i++)
    {
        std::cout << "### step: " << i << std::endl;
        test.train(inputs, outputs);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    test.print();
    return 0;
}
