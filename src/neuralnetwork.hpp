#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include <stdlib.h>
#include "neuron.hpp"

struct NeuCoor
{
    NeuCoor(){}
    NeuCoor(size_t layer, size_t neuron): layer(layer), neuron(neuron) {}
    size_t layer = 0;
    size_t neuron = 0;
};

class NeuralNetwork
{
    public:
        NeuralNetwork(std::vector<unsigned int> sizes);
        ~NeuralNetwork();
        bool connectNeurons(NeuCoor a, NeuCoor b); // a will be an input of b
        void setInput(const float &value, const size_t &key);
        void setInput(const std::vector<float> &values);
        std::vector<float> getOutputs();

        void initNetwork();
        void readyNetwork();
        bool train(std::vector<std::vector<float> >& inputs, std::vector<std::vector<float> >& outputs);
        void print();
    protected:
        std::vector<std::vector<Neuron*> > network;
        std::vector<float> inputs;
};

#endif // NEURALNETWORK_HPP
