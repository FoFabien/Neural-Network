#ifndef NEURON_HPP
#define NEURON_HPP

#include <stdlib.h>
#include <vector>

struct NeuInput
{
    NeuInput(){}
    NeuInput(void* ptr, double weight, bool isNeuron): ptr(ptr), weight(weight), isNeuron(isNeuron){}
    void* ptr = nullptr;
    double weight = 1.f;
    double delta = 0.f;
    double previous = 0.f;
    std::vector<double> sum_gradient;
    bool isNeuron = false;
};

class Neuron
{
    public:
        Neuron();
        virtual ~Neuron();
        void addInput(Neuron *ptr, double weight = 1.f);
        void addInput(double *ptr, double weight = 1.f);
        void addInput(NeuInput in);
        void delInput(void *ptr);
        std::vector<NeuInput>& getInputs();
        bool isConnected(Neuron* ptr) const;
        double getInputWeight(Neuron* ptr) const;

        void unready();
        bool isReady() const;
        void setDropout(const bool& b);
        bool isDropout() const;
        virtual double getOutput();
        virtual void doGradient(double node_delta);
        virtual void applyDelta(double learning_rate, double momentum, double weight_decay);

    protected:
        std::vector<NeuInput> inputs;
        double lastResult;
        bool ready;
        bool dropout;
};

#endif // NEURON_HPP
