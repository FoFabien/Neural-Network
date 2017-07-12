#include "neuralnetwork.hpp"
#include <iostream>
#include <random>
#include <chrono>

static std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());

NeuralNetwork::NeuralNetwork(std::vector<unsigned int> sizes)
{
    if(sizes.size() < 2)
        exit(0);
    inputs.resize(sizes[0], 0.f);
    for(auto &i: sizes)
    {
        if(i > 0)
        {
            network.push_back(std::vector<Neuron*>(i, new Neuron()));
        }
    }
    for(size_t i = 0; i < network[0].size(); ++i)
    {
        network[0][i]->addInput(&inputs[i]);
    }
}

NeuralNetwork::~NeuralNetwork()
{
    for(auto &i: network)
    {
        for(auto &j: i)
        {
            delete j;
        }
    }
}

bool NeuralNetwork::connectNeurons(NeuCoor a, NeuCoor b)
{
    if(a.layer >= network.size() || a.neuron >= network[a.layer].size() || b.layer >= network.size() || b.neuron >= network[b.layer].size() || a.layer >= b.layer)
        return false;
    if(a.layer == b.layer && a.neuron == b.neuron)
        return false;
    if(!network[b.layer][b.neuron]->isConnected(network[a.layer][a.neuron]))
        network[b.layer][b.neuron]->addInput(network[a.layer][a.neuron]);
    return true;
}

void NeuralNetwork::setInput(const float &value, const size_t &key)
{
    if(key < inputs.size())
        inputs[key] = value;
}

void NeuralNetwork::setInput(const std::vector<float> &values)
{
    for(size_t i = 0; i < values.size() && i < inputs.size(); ++i)
        inputs[i] = values[i];
}

std::vector<float> NeuralNetwork::getOutputs()
{
    std::vector<float> r;
    if(network.size() >= 2)
    {
        for(auto& i: network.back())
        {
            r.push_back(i->getOutput());
        }
    }
    return r;
}

void NeuralNetwork::initNetwork()
{
    std::normal_distribution<float> d(0.5, 1.f);
    for(size_t i = 1; i < network.size(); ++i)
    {
        network[i-1].push_back(new Neuron()); // bias neuron
        for(size_t j = 0; j < network[i].size(); ++j)
        {
            connectNeurons({i-1, network[i-1].size()-1}, {i, j}); // connect bias
            std::vector<NeuInput> inputs = network[i][j]->getInputs();
            for(auto& k: inputs)
            {
                k.weight = d(gen);
            }
        }
    }
}

void NeuralNetwork::readyNetwork()
{
    for(auto& i : network)
    {
        for(auto& j : i)
        {
            j->unready();
        }
    }
}

bool NeuralNetwork::train(std::vector<std::vector<float> >& inputs, std::vector<std::vector<float> >& outputs)
{
    if(inputs.size() != outputs.size())
        return false;

    for(size_t i = 0; i < inputs.size(); i++)
    {
        readyNetwork();
        setInput(inputs[i]);
        std::vector<float> res = getOutputs();
        std::vector<float> err;
        if(res.size() != outputs[i].size() || res.empty())
            return false;
        float mse = 0;
        std::cout << "Output(s) ";
        for(size_t j = 0; j < res.size(); j++)
        {
            std::cout << res[j] << " ";
            mse += (outputs[i][j] - res[j]) * (outputs[i][j] - res[j]);
            err.push_back(res[j] - outputs[i][j]);
        }
        std::cout << std::endl;
        mse /= res.size();
        std::cout << "mse = " << mse << std::endl;
        // end of batch
        // start of gradient
        float layer_delta = 0.f;
        float current_layer = 0.f;
        for(size_t j = network.size()-1; j > 0; --j)
        {
            current_layer = 0.f;
            for(size_t k = 0; (j == network.size()-1 && k < network[j].size())
                || (j < network.size()-1 && k < network[j].size()-1); ++k)
            {
                if(j == network.size()-1) // output layer
                {
                    float delta = (outputs[i][k] - res[k]) * network[j][k]->getOutputPrime();
                    current_layer += delta;
                    std::vector<NeuInput> &inputs = network[j][k]->getInputs();
                    for(auto& l: inputs)
                        l.gradient = delta * ((Neuron*)l.ptr)->getOutput();
                }
                else // hidden layer
                {
                    float sum_w = 0.f;
                    for(auto& l : network[j+1])
                    {
                        std::vector<NeuInput> &inputs = l->getInputs();
                        for(auto& m: inputs)
                            if(m.ptr == network[j][k])
                                sum_w += m.weight;
                    }
                    float delta = network[j][k]->getOutputPrime() * sum_w * layer_delta;
                    current_layer += delta;
                    std::vector<NeuInput> &inputs = network[j][k]->getInputs();
                    for(auto& l: inputs)
                        l.gradient = delta * ((Neuron*)l.ptr)->getOutput();
                }
            }
            layer_delta = current_layer;
        }
        // end of gradient calcul
        // start of backpropagation
        float learning_rate = 0.7f;
        float momentum = 0.3f;
        for(size_t j = 1; j < network.size(); ++j)
        {
            for(size_t k = 0; (j == network.size()-1 && k < network[j].size())
                || (j < network.size()-1 && k < network[j].size()-1); ++k)
            {
                std::vector<NeuInput> &inputs = network[j][k]->getInputs();
                for(auto& l: inputs)
                {
                    float delta_w = learning_rate * l.gradient + momentum * l.previous;
                    //std::cout << delta_w << " " << l.weight << " " << l.previous << std::endl;
                    l.weight += delta_w;
                    l.previous = delta_w;
                    //l.batch += l.weight + delta_w;
                }
            }
        }
        // end of backpropagation
    }

    return true;
}

void NeuralNetwork::print()
{
    std::cout << network.size() << " layer(s)" << std::endl;
    std::cout << " #0\t: " << network[0].size() << " neuron(s)" << std::endl;
    for(size_t i = 1; i < network.size(); ++i)
    {
        std::cout << " #" << i << "\t: " << network[i].size() << " neuron(s)" << std::endl;
        for(size_t j = 0; j < network[i].size(); ++j)
        {
            std::cout << "\tNeuron " << j << " :" << std::endl;
            std::vector<NeuInput> &inputs = network[i][j]->getInputs();
            for(auto& k: inputs)
            {
                std::cout << "\t\tInput weight: " << k.weight << std::endl;
            }
        }
    }
}
