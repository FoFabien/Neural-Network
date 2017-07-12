#include "neuron.hpp"
#include <cmath>

Neuron::Neuron()
{
    lastResult = 0.f;
    delta = 0.f;
    ready = false;
}

Neuron::~Neuron()
{

}

void Neuron::addInput(Neuron *ptr, float weight)
{
    if(ptr)
        addInput(NeuInput((void*)ptr, weight, true));
}

void Neuron::addInput(float *ptr, float weight)
{
    if(ptr)
        addInput(NeuInput((void*)ptr, weight, false));
}

void Neuron::addInput(NeuInput in)
{
    inputs.push_back(in);
}

std::vector<NeuInput>& Neuron::getInputs()
{
    return inputs;
}

bool Neuron::isConnected(Neuron* ptr) const
{
    for(auto &i: inputs)
        if(i.isNeuron && ptr == (Neuron*)i.ptr)
            return true;
    return false;
}

bool Neuron::isReady() const
{
    return ready;
}

float Neuron::getOutput()
{
    if(ready)
        return lastResult;
    float sum = 0;
    for(auto &i : inputs)
    {
        if(i.ptr == nullptr)
            continue;
        if(i.isNeuron)
        {
            sum += i.weight * ((Neuron*)i.ptr)->getOutput();
        }
        else
        {
            sum += i.weight * (*((float*)i.ptr));
        }
    }
    lastResult = 1 / (1 + exp(-sum));
    ready = true;
    return lastResult;
}

float Neuron::getOutputPrime()
{
    float sum = 0;
    for(auto &i : inputs)
    {
        if(i.ptr == nullptr)
            continue;
        if(i.isNeuron)
        {
            sum += i.weight * ((Neuron*)i.ptr)->getOutput();
        }
        else
        {
            sum += i.weight * (*((float*)i.ptr));
        }
    }
    float res = 1 / (1 + exp(-sum));
    res = res * (1 - res);
    return res;
}

void Neuron::unready()
{
    ready = false;
}


void Neuron::forward()
{

}

void Neuron::backward()
{

}
