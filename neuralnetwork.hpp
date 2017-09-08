#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include <stdlib.h>
#include <random>
#include <string>
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
        NeuralNetwork(); // only used when loading from file, you should use the other constructor for a brand new network
        NeuralNetwork(std::vector<size_t> sizes);
        ~NeuralNetwork();
        bool connectNeurons(NeuCoor a, NeuCoor b); // a will be an input of b
        void autoConnect();
        void setInput(const float &value, const size_t &key);
        void setInput(const std::vector<float> &values);
        void resizeLayer(size_t layer, size_t size);
        std::vector<float> getOutputs();
        float getWeight(NeuCoor a, NeuCoor b);

        void initNetwork();
        void readyNetwork();
        bool runTraining(std::vector<std::vector<float> > inputs, std::vector<std::vector<float> > outputs, size_t epochs, float learning_rate, bool print);
        bool train(const std::vector<std::vector<float> >& inputs, const std::vector<std::vector<float> >& outputs, float learning_rate, bool print);
        void print();
        bool load(const std::string &filename);
        bool save(const std::string &filename);
    protected:
        std::vector<std::vector<Neuron*> > network;
        std::vector<float*> inputs;

        std::mt19937 gen;
};

#endif // NEURALNETWORK_HPP
