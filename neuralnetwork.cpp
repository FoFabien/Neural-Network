#include "neuralnetwork.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <map>

NeuralNetwork::NeuralNetwork()
{
    gen = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
}

NeuralNetwork::NeuralNetwork(std::vector<size_t> sizes)
{
    if(sizes.size() < 2)
        exit(0);
    for(size_t i = 0; i < sizes[0]; ++i)
    {
        inputs.push_back(new float);
    }
    for(auto &i: sizes)
    {
        network.push_back(std::vector<Neuron*>());
        for(size_t j = 0; j < i; ++j)
            network.back().push_back(new Neuron());
    }
    for(size_t i = 0; i < network[0].size(); ++i)
    {
        network[0][i]->addInput(inputs[i]);
    }
    gen = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
}

NeuralNetwork::~NeuralNetwork()
{
    for(size_t i = 0; i < network.size(); ++i)
        for(size_t j = 0; j < network[i].size(); ++j)
            if(network[i][j])
                delete network[i][j];
    for(auto& i : inputs)
        delete i;
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
        *(inputs[key]) = value;
}

void NeuralNetwork::setInput(const std::vector<float> &values)
{
    for(size_t i = 0; i < values.size() && i < inputs.size(); ++i)
        *(inputs[i]) = values[i];
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

float NeuralNetwork::getWeight(NeuCoor a, NeuCoor b)
{
    if(a.layer >= network.size() || a.neuron >= network[a.layer].size() || b.layer >= network.size() || b.neuron >= network[b.layer].size() || a.layer >= b.layer)
        return false;
    if(a.layer == b.layer && a.neuron == b.neuron)
        return false;
    if(network[b.layer][b.neuron]->isConnected(network[a.layer][a.neuron]))
    {
        return network[b.layer][b.neuron]->getInputWeight(network[a.layer][a.neuron]);
    }
    return 0.f;
}

void NeuralNetwork::initNetwork()
{
    std::normal_distribution<float> d(0.5, 1.f);
    for(size_t i = 1; i < network.size(); ++i)
    {
        for(size_t j = 0; j < network[i].size(); ++j)
        {
            std::vector<NeuInput> &inputs = network[i][j]->getInputs();
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

bool NeuralNetwork::train(std::vector<std::vector<float> >& inputs, std::vector<std::vector<float> >& outputs, float learning_rate, size_t repeat, bool print)
{
    if(inputs.size() != outputs.size())
        return false;
    if(print)
        std::cout << "=== Results ====" << std::endl;

    for(size_t i = 0; i < inputs.size(); i++)
    {
        for(size_t s = 0; s < repeat; ++s)
        {
            readyNetwork();
            setInput(inputs[i]);
            std::vector<float> res = getOutputs();
            std::vector<float> err;
            if(res.size() != outputs[i].size() || res.empty())
                return false;

            if(print && s == repeat-1)
            {
                std::cout << "Input(s): ";
                for(float& n : inputs[i])
                    std::cout << n << " ";
                std::cout << " | Output(s): ";
                for(float& r : res)
                    std::cout << r << " ";
                std::cout << std::endl;
            }

            for(size_t j = network.size()-1; j > 0; --j)
            {
                for(size_t k = 0; k < network[j].size(); ++k)
                {
                    if(j == network.size()-1)
                    {
                        network[j][k]->doGradient(outputs[i][k], learning_rate);
                    }
                    else // improve for more hidden layers
                    {
                        float sum = 0;
                        for(size_t l = 0; l < network[j+1].size(); ++l)
                        {
                            float t = network[j+1][l]->getOutput();
                            sum += (t - outputs[i][l]) * t * (1 - t) * getWeight({j, k}, {j+1, l});
                        }
                        network[j][k]->doBackProp(sum, learning_rate);
                    }
                }
            }
            if(s % 100 == 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return true;
}

void NeuralNetwork::print()
{
    std::cout << network.size() << " layer(s)" << std::endl;
    for(size_t i = 0; i < network.size(); ++i)
    {
        std::cout << " #" << i << "\t: " << network[i].size() << " neuron(s)" << std::endl;
        for(size_t j = 0; j < network[i].size(); ++j)
        {
            std::cout << "\tNeuron " << j << " (addr=" << (int)(network[i][j]) << ") :" << std::endl;
            std::vector<NeuInput> &inputs = network[i][j]->getInputs();
            for(auto& k: inputs)
            {
                std::cout << "\t\tInput weight: " << k.weight;
                if(k.isNeuron)
                    std::cout << "\tto Neuron (addr=" << (int)k.ptr << ")" << std::endl;
                else
                    std::cout << "\tto Value (addr=" << (int)k.ptr << ")=" << (*(float*)(k.ptr)) << std::endl;
            }
        }
    }
}

#define FVERSION 0x00000000
bool NeuralNetwork::load(const std::string &filename)
{
    std::ifstream f(filename.c_str(), std::ios::in);
    if(!f)
        return false;
    std::map<size_t, Neuron*> list;
    size_t tmp;
    size_t last = 0;
    f >> tmp;
    if(tmp != FVERSION)
        return false;

    // clear
    for(size_t i = 0; i < network.size(); ++i)
    for(size_t j = 0; j < network[i].size(); ++j)
        if(network[i][j])
            delete network[i][j];
    for(auto& i : inputs)
        delete i;
    network.clear();
    inputs.clear();

    f >> tmp;
    for(size_t i = 0; i < tmp; ++i)
        network.push_back(std::vector<Neuron*>());

    f >> tmp;
    for(size_t i = 0; i < tmp; ++i)
        inputs.push_back(new float);


    for(auto& i : network)
    {
        f >> tmp;
        for(size_t j = 0; j < tmp; ++j)
            i.push_back(new Neuron());

        for(auto& j : i)
        {
            f >> tmp;
            list[tmp] = j;
            size_t input_count;
            f >> input_count;
            for(size_t k = 0; k < input_count; ++k)
            {
                f >> tmp;
                if(tmp == 1)
                {
                    float w;
                    f >> tmp >> w;
                    j->addInput(list[tmp], w);
                }
                else
                {
                    j->addInput(inputs[last]);
                    ++last;
                }
            }
        }
    }

    return true;
}

bool NeuralNetwork::save(const std::string &filename)
{
    std::ofstream f(filename.c_str(), std::ios::out | std::ios::trunc);
    if(!f)
        return false;
    std::map<Neuron*, size_t> list;
    size_t last = 0;
    f << (size_t)FVERSION << " " << network.size() << " " << inputs.size() << " ";
    for(auto& i : network)
    {
        f << i.size() << " ";
        for(auto& j : i)
        {
            if(list.find(j) == list.end())
            {
                list[j] = last;
                ++last;
            }
            f << list[j] << " ";
            std::vector<NeuInput>& in = j->getInputs();
            f << in.size() << " ";
            for(auto& k : in)
            {
                if(k.isNeuron)
                {
                    f << (size_t)1 << " " << list[(Neuron*)k.ptr] << " " << k.weight << " ";
                }
                else
                {
                    f << (size_t)0 << " ";
                }
            }
        }
    }
    return true;
}
