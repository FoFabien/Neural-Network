#include "neuron.hpp"
#include <cmath>
#include <iostream>

#define M_PI_2        1.57079632679489661923	/* pi/2 */
#define M_PI_2_INV    (1.0/M_PI_2)
#define M_2_SQRTPI    1.12837916709551257390    /* 2/sqrt(pi) */
#define ERF_COEF      (1.0/M_2_SQRTPI)

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

void Neuron::delInput(void *ptr)
{
    for(std::vector<NeuInput>::iterator it = inputs.begin(); it != inputs.end(); ++it)
    {
        if(it->ptr == ptr)
        {
            inputs.erase(it);
            return;
        }
    }
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

void Neuron::doGradient(float sum, float step)
{
    if(!ready)
        getOutput();
    for(auto& i : inputs)
    {
        if(i.isNeuron)
            i.weight = i.weight - step * sum * lastResult * (1 - lastResult) * ((Neuron*)i.ptr)->getOutput();
    }
}

void Neuron::unready()
{
    ready = false;
}
