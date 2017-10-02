#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include <stdlib.h>
#include <random>
#include <string>
#include "neuron.hpp"

#include <mutex>

struct NeuCoor
{
    NeuCoor(){}
    NeuCoor(size_t layer, size_t neuron): layer(layer), neuron(neuron) {}
    size_t layer = 0;
    size_t neuron = 0;
};

class NeuralNetwork
{
    public:
        NeuralNetwork(); // only used when loading from file, you should use the other constructor for a brand new network
        NeuralNetwork(std::vector<size_t> sizes);
        NeuralNetwork(std::vector<std::vector<Neuron*> > &net);
        ~NeuralNetwork();
        void clear();
        bool connectNeurons(NeuCoor a, NeuCoor b); // a will be an input of b
        void autoConnect();
        void setInput(const double &value, const size_t &key);
        void setInput(const std::vector<double> &values);
        void resizeLayer(size_t layer, size_t size);
        void calcOutputs();
        const std::vector<double>& getOutputs();
        std::vector<double> getOutputsCpy();
        double getWeight(NeuCoor a, NeuCoor b);
        void randomDropout();
        void clearDropout();

        void initNetwork();
        void readyNetwork();
        void setTrainingSleepRate(int r);
        void pauseTraining();
        void resumeTraining();
        void stopTraining();
        bool runTraining(const std::vector<std::vector<double> > &inputs, const std::vector<std::vector<double> > &outputs, const size_t &epochs, const double &learning_rate, const double &momentum, const double &weight_decay, const bool &dropout, const std::string& save_file = "");
        bool train(const std::vector<std::vector<double> >& inputs, const std::vector<std::vector<double> >& outputs, const double &learning_rate, const double &momentum, const double &weight_decay, const bool &dropout);
        void print() const;
        void printTraining() const;
        bool load(const std::string &filename);
        bool save(const std::string &filename);

        std::vector<std::vector<Neuron*> > stealNetwork();
        void setNetwork(std::vector<std::vector<Neuron*> > &net);
        static NeuralNetwork* merge(NeuralNetwork& A, NeuralNetwork& B);

    protected:
        std::vector<std::vector<Neuron*> > network;
        std::vector<double> in;
        std::vector<double> out;

        size_t train_percent;
        size_t train_epoch;
        bool training;
        bool train_paused;
        bool train_stop;
        int train_sleep_counter;
        std::mutex mutex;
        std::mt19937 gen;
};

#endif // NEURALNETWORK_HPP
