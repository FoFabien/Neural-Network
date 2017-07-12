#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>

struct NeuInput
{
    NeuInput(){}
    NeuInput(void* ptr, float weight, bool isNeuron): ptr(ptr), weight(weight), isNeuron(isNeuron){}
    void* ptr = nullptr;
    float weight = 1.f;
    float gradient = 0.f;
    float previous = 0.f;
    float batch = 0.f;
    bool isNeuron = false;
};

class Neuron
{
    public:
        Neuron();
        ~Neuron();
        void addInput(Neuron *ptr, float weight = 1.f);
        void addInput(float *ptr, float weight = 1.f);
        void addInput(NeuInput in);
        std::vector<NeuInput>& getInputs();
        bool isConnected(Neuron* ptr) const;

        bool isReady() const;
        float getOutput();
        float getOutputPrime();
        void unready();

    protected:
        std::vector<NeuInput> inputs;
        float lastResult;
        float delta;
        bool ready;
};

#endif // NEURON_HPP
