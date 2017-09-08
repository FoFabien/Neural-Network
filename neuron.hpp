#ifndef NEURON_HPP
#define NEURON_HPP

#include <stdlib.h>
#include <vector>

struct NeuInput
{
    NeuInput(){}
    NeuInput(void* ptr, float weight, bool isNeuron): ptr(ptr), weight(weight), isNeuron(isNeuron){}
    void* ptr = nullptr;
    float weight = 1.f;
    bool isNeuron = false;
};

class Neuron
{
    public:
        Neuron();
        virtual ~Neuron();
        void addInput(Neuron *ptr, float weight = 1.f);
        void addInput(float *ptr, float weight = 1.f);
        void addInput(NeuInput in);
        void delInput(void *ptr);
        std::vector<NeuInput>& getInputs();
        bool isConnected(Neuron* ptr) const;
        float getInputWeight(Neuron* ptr) const;

        void unready();
        bool isReady() const;
        virtual float getOutput();
        void doGradient(float sum, float step = 0.01);

    protected:
        std::vector<NeuInput> inputs;
        float lastResult;
        float delta;
        float sum;
        bool ready;
};

#endif // NEURON_HPP
