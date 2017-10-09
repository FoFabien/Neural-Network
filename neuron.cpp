#include "neuron.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>

#include "kahan.hpp"

Neuron::Neuron()
{
    lastResult = 0.f;
    bias = 0.f;
    bias_delta = 0.f;
    node_delta = 0.f;
    ready = false;
    dropout = false;
}

Neuron::~Neuron()
{

}

void Neuron::addInput(const NeuLink &in)
{
    inputs.push_back(in);
}

void Neuron::addOutput(const NeuLink &out)
{
    outputs.push_back(out);
}

void Neuron::delInput(void *ptr)
{
    for(std::vector<NeuLink>::iterator it = inputs.begin(); it != inputs.end(); ++it)
    {
        if(it->input.ptr == ptr)
        {
            if(it->input.isNeuron)
                ((Neuron*)it->input.ptr)->delOutput(this);
            inputs.erase(it);
            return;
        }
    }
}

void Neuron::delOutput(void *ptr)
{
    for(std::vector<NeuLink>::iterator it = outputs.begin(); it != outputs.end(); ++it)
    {
        if(it->output.ptr == ptr)
        {
            outputs.erase(it);
            return;
        }
    }
}

std::vector<NeuLink>& Neuron::getInputLinks()
{
    return inputs;
}

std::vector<NeuLink>& Neuron::getOutputLinks()
{
    return outputs;
}

bool Neuron::isInputOf(Neuron* ptr) const
{
    if(dropout) // dropout aren't considered as connected
        return false;
    for(auto &i: outputs)
        if(i.output.isNeuron && ptr == (Neuron*)i.output.ptr)
            return true;
    return false;
}

bool Neuron::isOutputOf(Neuron* ptr) const
{
    if(dropout) // dropout aren't considered as connected
        return false;
    for(auto &i: inputs)
        if(i.input.isNeuron && ptr == (Neuron*)i.input.ptr)
            return true;
    return false;
}

double Neuron::getInputWeight(Neuron* ptr) const
{
    for(auto &i: inputs)
        if(i.input.isNeuron && ptr == (Neuron*)i.input.ptr)
            return i.data->weight;
    return 0.f;
}

void Neuron::setBias(const double& b)
{
    bias = b;
}

double Neuron::getBias() const
{
    return bias;
}

void Neuron::setNodeDelta(const double& d)
{
    node_delta = d;
}

double Neuron::getNodeDelta() const
{
    return node_delta;
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
        void* n = i.input.ptr;
        if(!n)
            continue;
        if(i.input.isNeuron)
        {
            if(((Neuron*)n)->isDropout())
                continue;
            in = ((Neuron*)n)->getOutput();
        }
        else // basically, input layer = just return the value (the neuron should have a single input in this case)
            in = (*((double*)n));
        to_sum.push_back(i.data->weight * in);
    }
    to_sum.push_back(bias);
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
        void* n = i.input.ptr;
        if(i.input.isNeuron && !((Neuron*)n)->isDropout())
            i.data->sum_gradient.push_back(node_delta * ((Neuron*)n)->getOutput()); // gradient
    }
    bias_gradient.push_back(node_delta);
}

void Neuron::applyDelta(double learning_rate, double momentum, double weight_decay)
{
    if(dropout)
        return;
    if(!ready)
        return; // it means you shouldn't call this function
    double previous;
    for(auto& i : inputs)
    {
        void* n = i.input.ptr;
        if(i.input.isNeuron && !((Neuron*)n)->isDropout())
        {
            std::sort(i.data->sum_gradient.begin(), i.data->sum_gradient.end(), std::greater<double>());
            previous = i.data->delta;
            i.data->delta = - learning_rate * DoubleSum(i.data->sum_gradient) + momentum * previous - i.data->weight * weight_decay;
            i.data->weight += i.data->delta;
            i.data->sum_gradient.clear();
        }
    }
    std::sort(bias_gradient.begin(), bias_gradient.end(), std::greater<double>());
    previous = bias_delta;
    bias_delta = - learning_rate * DoubleSum(bias_gradient) + momentum * previous - bias * weight_decay;
    bias += bias_delta;
    bias_gradient.clear();
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
