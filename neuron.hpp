#ifndef NEURON_HPP
#define NEURON_HPP

#include <stdlib.h>
#include <vector>
#include <memory>

struct NeuItem // store a pointer to either a neuron or an input value
{
    NeuItem(){}
    NeuItem(void* ptr, bool isNeuron): ptr(ptr), isNeuron(isNeuron){}
    void* ptr = nullptr;
    bool isNeuron = false;
};

struct NeuWeight // store a weight and the variables for training it
{
    double weight = 1.f;
    double delta = 0.f;
    std::vector<double> sum_gradient;
};

struct NeuLink // link between two neurons
{
    NeuLink(): data(new NeuWeight()){}
    NeuLink(NeuItem input, NeuItem output): input(input), output(output), data(new NeuWeight()){}
    NeuLink(NeuItem input, NeuItem output, double weight): input(input), output(output), data(new NeuWeight())
    {
        data->weight = weight;
    }
    NeuItem input;
    NeuItem output;
    std::shared_ptr<NeuWeight> data;
};

class Neuron
{
    public:
        Neuron();
        virtual ~Neuron();
        void addInput(const NeuLink &in); // add an input to the neuron
        void addOutput(const NeuLink &out); // add an output to the neuron
        void delInput(void *ptr); // delete an input. Will also delete the ouput on the linked neuron
        void delOutput(void *ptr); // shouldn't be called but it exists
        std::vector<NeuLink>& getInputLinks(); // input list
        std::vector<NeuLink>& getOutputLinks(); // output list
        bool isInputOf(Neuron* ptr) const; // true if is an input of ptr
        bool isOutputOf(Neuron* ptr) const; // true if is an output of ptr
        double getInputWeight(Neuron* ptr) const; // return the weight, 0 if ptr isn't an input
        void setBias(const double& b); // set the bias value
        double getBias() const; // return the bias value
        void setNodeDelta(const double& d); // set the node delta value used for training
        double getNodeDelta() const; // return the node delta value

        void unready(); // set the neuron as not ready. Output will be calculated from scratch
        bool isReady() const; // true if ready, meaning the output is already calculated
        void setInputNeuron(const bool& b); // set the neuron in input mode
        bool isInputNeuron() const;
        void setDropout(const bool& b); // set the dropout state
        bool isDropout() const;
        double getOutput(); // return the output value
        void doGradient(const double &node_delta); // finish the gradient calculation
        void applyDelta(const double &learning_rate, const double &momentum, const double &weight_decay); // modify the weights (to be called after the batch is over)

    protected:
        std::vector<NeuLink> inputs;
        std::vector<NeuLink> outputs;
        double lastResult; // output value

        // bias variable, it works like a weight
        double bias;
        double bias_delta;
        std::vector<double> bias_gradient;

        double node_delta; // used for training

        bool ready; // true if lastResult contains the output
        bool isInput; // true if it's an input neuron
        bool dropout; // true to disable the neuron
};

#endif // NEURON_HPP
