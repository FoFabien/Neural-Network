#include "neuron.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>

#include "kahan.hpp"

#define M_PI_2        1.57079632679489661923	/* pi/2 */
#define M_PI_2_INV    (1.0/M_PI_2)
#define M_2_SQRTPI    1.12837916709551257390    /* 2/sqrt(pi) */
#define ERF_COEF      (1.0/M_2_SQRTPI)

Neuron::Neuron()
{
    lastResult = 0.f;
    ready = false;
    dropout = false;
}

Neuron::~Neuron()
{

}

void Neuron::addInput(Neuron *ptr, double weight)
{
    if(ptr)
        addInput(NeuInput((void*)ptr, weight, true));
}

void Neuron::addInput(double *ptr, double weight)
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
    if(dropout) // dropout aren't considered as connected
        return false;
    for(auto &i: inputs)
        if(i.isNeuron && ptr == (Neuron*)i.ptr)
            return true;
    return false;
}

double Neuron::getInputWeight(Neuron* ptr) const
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

double Neuron::getOutput()
{
    if(ready)
        return lastResult;
    double sum = 0;
    double in = 0;
    std::vector<double> to_sum;
    for(auto &i : inputs)
    {
        if(i.ptr == nullptr)
            continue;
        if(i.isNeuron)
        {
            if(((Neuron*)i.ptr)->isDropout())
                continue;
            in = ((Neuron*)i.ptr)->getOutput();
        }
        else
            in = (*((double*)i.ptr));
        to_sum.push_back(i.weight * in);
    }
    ready = true;
    if(dropout)
    {
        lastResult = 0;
        return 0;
    }
    std::sort(to_sum.begin(), to_sum.end(), std::greater<double>());
    sum = DoubleSum(to_sum); // to avoid a loss of precision during the sum
    lastResult = 1 / (1 + exp(-sum));

    return lastResult;
}

void Neuron::doGradient(double node_delta)
{
    if(dropout)
        return;
    if(!ready)
        return; // it means you shouldn't call this function
    for(auto& i : inputs)
    {
        if(i.isNeuron && !((Neuron*)i.ptr)->isDropout())
            i.sum_gradient.push_back(node_delta * ((Neuron*)i.ptr)->getOutput()); // gradient
    }
}

void Neuron::applyDelta(double learning_rate, double momentum, double weight_decay)
{
    if(dropout)
        return;
    if(!ready)
        return; // it means you shouldn't call this function
    for(auto& i : inputs)
    {
        if(i.isNeuron && !((Neuron*)i.ptr)->isDropout())
        {
            std::sort(i.sum_gradient.begin(), i.sum_gradient.end(), std::greater<double>());
            double tmp = i.delta;
            i.delta = - learning_rate * DoubleSum(i.sum_gradient) + momentum * i.previous - i.weight * weight_decay; // loss function
            i.previous = tmp;
            i.weight += i.delta;
            i.sum_gradient.clear();
        }
    }
}

void Neuron::unready()
{
    ready = false;
}

void Neuron::setDropout(const bool& b)
{
    dropout = b;
}

bool Neuron::isDropout() const
{
    return dropout;
}
