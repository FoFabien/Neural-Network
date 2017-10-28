#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include <stdlib.h>
#include <random>
#include <string>
#include "neuron.hpp"

#include <mutex>

struct NeuCoor // rarely used to point at a specific neuron in the network (basically, X and Y coordinates)
{
    NeuCoor(){}
    NeuCoor(size_t layer, size_t neuron): layer(layer), neuron(neuron) {}
    size_t layer = 0;
    size_t neuron = 0;
};

// Note : Neurons from the first layer must have a single input. The input mustn't be another neuron

class NeuralNetwork
{
    public:
        NeuralNetwork(); // only used when loading from file, you should use the other constructor for a brand new network
        NeuralNetwork(std::vector<size_t> sizes); // create a network with a specific size ({3, 2, 1} = 3 layers with 3 neurons on the first layer, 2 on the second and 1 on the last)
        NeuralNetwork(std::vector<std::vector<Neuron*> > &net); // use an existing network
        ~NeuralNetwork();
        void clear();
        bool connectNeurons(NeuCoor a, NeuCoor b); // a will be an input of b
        void autoConnect(); // connect all the neurons from a layer to those of the next one
        void setInput(std::vector<double> &values); // set the input values. The vector MUST continue to exist until you got your output
        void resizeLayer(size_t layer, size_t size); // resize a layer (note: slighty EXPERIMENTAL)
        void calcOutputs(); // calculate the output according to the inputs, results are stored in out
        const std::vector<double>& getOutputs(); // return the out vector
        std::vector<double> getOutputsCpy(); // return a copy of the same vector
        double getWeight(NeuCoor a, NeuCoor b); // return the weight between neuron A and neuron B (assuming they are connected, 0 otherwise)
        void randomDropout(); // randomly create dropout in the network (EXPERIMENTAL)
        void clearDropout(); // clear the dropout states

        void initNetwork(); // randomly initialize all weights and biases
        void readyNetwork(); // ready the network to calculate the next ouputs
        void setTrainingSleepRate(int r); // training sleep rate : <2 = no sleep, 2= maximum sleep rate, >2 bigger equals less sleep time
        void pauseTraining();
        void resumeTraining();
        void stopTraining();
        size_t trainingState() const;
        bool runTraining(std::vector<std::vector<double> > &inputs, const std::vector<std::vector<double> > &outputs, const size_t &epochs, const size_t &batch_size, const double &learning_rate, const double &momentum, const double &weight_decay, const bool &dropout,
                        const std::string& save_file); // will call train a certain number of time according to epochs
        bool train(std::vector<std::vector<double> >& inputs, const std::vector<std::vector<double> >& outputs, const size_t &batch_size, const double &learning_rate, const double &momentum, const double &weight_decay, const bool &dropout); // train the network with the batch of data
        void print(const bool &detail = true) const; // print the network
        void printTraining() const; // print the training state
        bool load(const std::string &filename);
        bool save(const std::string &filename);

        std::vector<std::vector<Neuron*> > stealNetwork(); // remove and return the network
        void setNetwork(std::vector<std::vector<Neuron*> > &net); // set the network with a new one
        static NeuralNetwork* merge(NeuralNetwork& A, NeuralNetwork& B); // merge two network by stealing them and return the result

    protected:
        std::vector<std::vector<Neuron*> > network; // neural network
        //std::vector<double> in;
        std::vector<double> out; // store the outputs

        // training variables
        size_t train_percent;
        size_t train_epoch;
        bool training;
        bool train_paused;
        bool train_stop;
        int train_sleep_counter;

        std::mutex mutex; // for multithreading
        std::mt19937 gen; // for random stuff
};

#endif // NEURALNETWORK_HPP
