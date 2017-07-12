#include "neuron.hpp"
#include <cmath>
#include <iostream>

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

float Neuron::getInputWeight(Neuron* ptr) const
{
    for(auto &i: inputs)
        if(i.isNeuron && ptr == (Neuron*)i.ptr)
            return i.weight;
    return 0.f;
}

bool Neuron::isReady() const
{
    return ready;
}

float Neuron::getSum()
{
    if(!ready)
        getOutput();
    return sum;
}

float Neuron::getOutput()
{
    if(ready)
        return lastResult;
    sum = 0;
    float in = 0;
    for(auto &i : inputs)
    {
        if(i.ptr == nullptr)
            continue;
        if(i.isNeuron)
            in = ((Neuron*)i.ptr)->getOutput();
        else
            in = (*((float*)i.ptr));
        sum += i.weight * in;
    }
    lastResult = 1 / (1 + exp(-sum));
    ready = true;
    return lastResult;
}

float Neuron::getOutputPrime()
{
    if(!ready)
        getOutput();
    return lastResult * (1 - lastResult);
}

float Neuron::getSquaredLoss(float actual)
{
    if(!ready)
        getOutput();
    return 0.5 * (actual - lastResult) * (actual - lastResult);
}

void Neuron::doGradient(float actual, float step)
{
    if(!ready)
        getOutput();
    for(auto& i : inputs)
    {
        if(i.isNeuron)
            i.weight = i.weight - step * (lastResult - actual) * lastResult * (1 - lastResult) * ((Neuron*)i.ptr)->getOutput();
        /*else
            i.weight = i.weight - step * (lastResult - actual) * lastResult * (1 - lastResult) * (*(float*)i.ptr);*/
    }
}

void Neuron::doBackProp(float sum, float step)
{
    if(!ready)
        getOutput();
    for(auto& i : inputs)
    {
        if(i.isNeuron)
            i.weight = i.weight - step * sum * lastResult * (1 - lastResult) * ((Neuron*)i.ptr)->getOutput();
        /*else
            i.weight = i.weight - step * sum * lastResult * (1 - lastResult) * (*(float*)i.ptr);*/
    }
}

void Neuron::unready()
{
    ready = false;
}
