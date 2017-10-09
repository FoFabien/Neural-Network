#ifndef NEURON_HPP
#define NEURON_HPP

#include <stdlib.h>
#include <vector>
#include <memory>

struct NeuItem
{
    NeuItem(){}
    NeuItem(void* ptr, bool isNeuron): ptr(ptr), isNeuron(isNeuron){}
    void* ptr = nullptr;
    bool isNeuron = false;
};

struct NeuWeight
{
    double weight = 1.f;
    double delta = 0.f;
    std::vector<double> sum_gradient;
};

struct NeuLink
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
        void addInput(const NeuLink &in);
        void addOutput(const NeuLink &out);
        void delInput(void *ptr);
        void delOutput(void *ptr);
        std::vector<NeuLink>& getInputLinks();
        std::vector<NeuLink>& getOutputLinks();
        bool isInputOf(Neuron* ptr) const;
        bool isOutputOf(Neuron* ptr) const;
        double getInputWeight(Neuron* ptr) const;
        void setBias(const double& b);
        double getBias() const;
        void setNodeDelta(const double& d);
        double getNodeDelta() const;

        void unready();
        bool isReady() const;
        void setDropout(const bool& b);
        bool isDropout() const;
        double getOutput();
        void doGradient(double node_delta);
        void applyDelta(double learning_rate, double momentum, double weight_decay);

    protected:
        std::vector<NeuLink> inputs;
        std::vector<NeuLink> outputs;
        double lastResult;
        double bias;
        double bias_delta;
        std::vector<double> bias_gradient;
        double node_delta;
        bool ready;
        bool dropout;
};

#endif // NEURON_HPP
