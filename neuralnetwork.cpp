#include "neuralnetwork.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <map>
#include <algorithm>
#include <set>

#include "kahan.hpp"
#include "misc.hpp"

NeuralNetwork::NeuralNetwork()
{
    train_percent = 0;
    train_epoch = 0;
    training = false;
    train_paused = false;
    train_stop = false;
    train_sleep_counter = 0;
    gen = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
}

NeuralNetwork::NeuralNetwork(std::vector<size_t> sizes)
{
    if(sizes.size() < 2)
        return;
    for(size_t i = 0; i < sizes[0]; ++i)
        in.push_back(0);
    for(auto &i: sizes)
    {
        network.push_back(std::vector<Neuron*>());
        for(size_t j = 0; j < i; ++j)
            network.back().push_back(new Neuron());
    }
    for(size_t i = 0; i < network[0].size(); ++i)
        network[0][i]->addInput(&in[i]);
    for(size_t i = 0; i < network.back().size(); ++i)
        out.push_back(0);

    train_percent = 0;
    train_epoch = 0;
    training = false;
    train_paused = false;
    train_stop = false;
    train_sleep_counter = 0;
    gen = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
}

NeuralNetwork::NeuralNetwork(std::vector<std::vector<Neuron*> > &net)
{
    network = net;
    if(network.size() < 2)
        return;
    for(size_t i = 0; i < network[0].size(); ++i)
        in.push_back(0);
    for(size_t i = 0; i < network.back().size(); ++i)
        out.push_back(0);

    train_percent = 0;
    train_epoch = 0;
    training = false;
    train_paused = false;
    train_stop = false;
    train_sleep_counter = 0;
    gen = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
}

NeuralNetwork::~NeuralNetwork()
{
    clear();
}

void NeuralNetwork::clear()
{
    for(size_t i = 0; i < network.size(); ++i)
        for(size_t j = 0; j < network[i].size(); ++j)
            if(network[i][j])
                delete network[i][j];
    in.clear();
    out.clear();
    train_percent = 0;
    train_epoch = 0;
    training = false;
    train_paused = false;
    train_stop = false;
}

bool NeuralNetwork::connectNeurons(NeuCoor a, NeuCoor b)
{
    if(a.layer >= network.size() || a.neuron >= network[a.layer].size() || b.layer >= network.size() || b.neuron >= network[b.layer].size() || a.layer >= b.layer)
        return false;
    if(a.layer == b.layer && a.neuron == b.neuron)
        return false;
    if(!network[b.layer][b.neuron]->isConnected(network[a.layer][a.neuron]))
        network[b.layer][b.neuron]->addInput(network[a.layer][a.neuron]);
    return true;
}

void NeuralNetwork::autoConnect()
{
    for(size_t i = 1; i < network.size(); ++i)
    {
        for(size_t j = 0; j < network[i].size(); ++j)
        {
            for(size_t k = 0; k < network[i-1].size(); ++k)
            {
                connectNeurons(NeuCoor(i-1, k), NeuCoor(i, j));
            }
        }
    }
}

void NeuralNetwork::setInput(const double &value, const size_t &key)
{
    if(key < in.size())
        in[key] = value;
}

void NeuralNetwork::setInput(const std::vector<double> &values)
{
    for(size_t i = 0; i < values.size() && i < in.size(); ++i)
        in[i] = values[i];
}

void NeuralNetwork::resizeLayer(size_t layer, size_t size)
{
    if(size == 0)
        return;
    std::normal_distribution<double> d(0.5, 1.f);
    if(layer < network.size()-1 && layer > 0)
    {
        while(network[layer].size() < size)
        {
            network[layer].push_back(new Neuron());
            for(size_t i = 0; i < network[layer-1].size(); ++i)
                network[layer].back()->addInput(network[layer-1][i], d(gen));
            for(size_t i = 0; layer+1 < network.size() && i < network[layer+1].size(); ++i)
                network[layer+1][i]->addInput(network[layer].back(), d(gen));
        }
        while(network[layer].size() > size)
        {
            for(size_t i = 0; layer+1 < network.size() && i < network[layer+1].size(); ++i)
                network[layer+1][i]->delInput(network[layer].back());
            delete network[layer].back();
            network[layer].pop_back();
        }
        std::cout << network[layer].size() << std::endl;
    }
}

void NeuralNetwork::calcOutputs()
{
    if(network.size() >= 2)
    {
        for(size_t i = 0; i < out.size() && i < network.back().size(); ++i)
            out[i] = network.back()[i]->getOutput();
    }
}

const std::vector<double>& NeuralNetwork::getOutputs()
{
    calcOutputs();
    return out;
}

std::vector<double> NeuralNetwork::getOutputsCpy()
{
    calcOutputs();
    return out;
}

double NeuralNetwork::getWeight(NeuCoor a, NeuCoor b)
{
    if(a.layer >= network.size() || a.neuron >= network[a.layer].size() || b.layer >= network.size() || b.neuron >= network[b.layer].size() || a.layer >= b.layer)
        return false;
    if(a.layer == b.layer && a.neuron == b.neuron)
        return false;
    if(network[b.layer][b.neuron]->isConnected(network[a.layer][a.neuron]))
    {
        return network[b.layer][b.neuron]->getInputWeight(network[a.layer][a.neuron]);
    }
    return 0.f;
}

void NeuralNetwork::initNetwork()
{
    std::normal_distribution<double> d(0.5, 1.f);
    for(size_t i = 1; i < network.size(); ++i)
    {
        for(size_t j = 0; j < network[i].size(); ++j)
        {
            network[i][j]->setBias(d(gen));
            std::vector<NeuInput> &inputs = network[i][j]->getInputs();
            for(auto& k: inputs)
            {
                k.weight = d(gen);
            }
        }
    }
}

void NeuralNetwork::readyNetwork()
{
    for(auto& i : network)
    {
        for(auto& j : i)
        {
            j->unready();
        }
    }
}

void NeuralNetwork::setTrainingSleepRate(int r)
{
    train_sleep_counter = r;
}


void NeuralNetwork::pauseTraining()
{
    train_paused = true;
}

void NeuralNetwork::resumeTraining()
{
    train_paused = false;
}

void NeuralNetwork::stopTraining()
{
    train_stop = false;
}

bool NeuralNetwork::runTraining(const std::vector<std::vector<double> > &inputs, const std::vector<std::vector<double> > &outputs, const size_t &epochs, const double &learning_rate, const double& momentum, const double &weight_decay,const  bool &dropout, const std::string& save_file)
{
    if(inputs.size() != outputs.size() || inputs.empty())
        return false;

    size_t step = epochs / 10;
    if(epochs < 10)
        step = epochs;

    train_percent = 0;
    training = true;
    train_stop = false;

    // number of time the dataset is used
    for(train_epoch = 0; train_epoch < epochs; train_epoch++)
    {
        if((train_epoch+1) % step == 0 && save_file != "")
        {
            if(save(save_file))
                std::cout << "Saved to " << save_file << std::endl;
            else
                std::cout << "Failed to save to " << save_file << std::endl;
        }

        if(!train(inputs, outputs, learning_rate, momentum, weight_decay, dropout))
        {
            std::cout << "error at epoch " << train_epoch << std::endl;
            break;
        }
        train_percent = (train_epoch * 100) / epochs;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        while(train_paused && !train_stop)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if(train_stop)
            break;
    }
    train_stop = false;
    train_paused = false;
    training = false;
    return true;
}

bool NeuralNetwork::train(const std::vector<std::vector<double> >& inputs, const std::vector<std::vector<double> >& outputs, const double &learning_rate, const double &momentum, const double &weight_decay, const bool &dropout)
{
    mutex.lock();
    if(inputs.size() != outputs.size() || inputs.empty())
    {
        std::cout << "dataset size error" << std::endl;
        mutex.unlock();
        return false;
    }

    int sleep_counter = 0;
    std::vector<double> vec1, vec2; // to contain stuff

    clearDropout();
    if(dropout)
        randomDropout();
    for(size_t i = 0; i < inputs.size(); ++i)
    {
        readyNetwork();
        setInput(inputs[i]);
        calcOutputs();

        if(out.size() != outputs[i].size() || out.empty())
        {
            std::cout << "output size error" << std::endl;
            mutex.unlock();
            return false;
        }

        // back propagation
        double node_delta;
        std::vector<double> sum;
        for(size_t j = network.size()-1; j > 0; --j)
        {
            for(size_t k = 0; k < network[j].size(); ++k)
            {
                if(j == network.size()-1)
                {
                    node_delta = (out[k] - outputs[i][k]) /*error*/ * out[k] * (1 - out[k]) /*derivative*/;
                }
                else
                {
                    double tmp = network[j][k]->getOutput();
                    sum.clear();
                    if(!network[j][k]->isDropout())
                    {
                        for(size_t l = 0; l < network[j+1].size(); ++l)
                        {
                            if(network[j+1][l]->isConnected(network[j][k]))
                            {
                                sum.push_back(network[j+1][l]->getInputWeight(network[j][k]) * vec2[l]);
                            }
                        }
                        std::sort(sum.begin(), sum.end(), std::greater<double>());
                    }
                    node_delta = tmp * (1 - tmp) /*derivative*/ * DoubleSum(sum) /*sum weight*node_delta of next layer*/;
                }
                network[j][k]->doGradient(node_delta);
                vec1.push_back(node_delta);
            }
            vec2 = std::move(vec1);
        }
        ++sleep_counter;
        if(train_sleep_counter > 1 && sleep_counter % train_sleep_counter == 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    for(size_t i = 1; i < network.size(); ++i)
        for(size_t j = 0; j < network[i].size(); ++j)
            network[i][j]->applyDelta(learning_rate, momentum, weight_decay);
    mutex.unlock();
    return true;
}

void NeuralNetwork::print(const bool &detail) const
{
    std::cout << network.size() << " layer(s)" << std::endl;
    for(size_t i = 0; i < network.size(); ++i)
    {
        std::cout << " #" << i << "\t: " << network[i].size() << " neuron(s)" << std::endl;
        for(size_t j = 0; j < network[i].size(); ++j)
        {
            std::cout << "\tNeuron " << j << " (addr=" << (int)(network[i][j]) << ") (bias=" << network[i][j]->getBias() << ") :" << std::endl;
            if(!detail) continue;
            std::vector<NeuInput> &inputs = network[i][j]->getInputs();
            for(auto& k: inputs)
            {
                std::cout << "\t\tInput weight: " << k.weight;
                if(k.isNeuron)
                    std::cout << "\tto Neuron (addr=" << (int)k.ptr << ")" << std::endl;
                else
                    std::cout << "\tto Value (addr=" << (int)k.ptr << ")=" << (*(double*)(k.ptr)) << std::endl;
            }
        }
    }
}

void NeuralNetwork::printTraining() const
{
    std::cout << "Network " << ((training == true) ? ("is training") : ("isn't in training")) << std::endl;
    if(training)
    {
        std::cout << "Status: epoch " << train_epoch << " (" << train_percent << "%)" << std::endl;
        std::cout << "Pause: " << ((train_paused == true) ? ("True") : ("False")) << std::endl;
        std::cout << "Sleep rate: " << train_sleep_counter << std::endl;
    }
}

#define FVERSION 4
bool NeuralNetwork::load(const std::string &filename)
{
    mutex.lock();
    std::ifstream f(filename.c_str(), std::ios::in);
    if(!f)
    {
        mutex.unlock();
        return false;
    }
    std::map<size_t, Neuron*> list;
    size_t tmp;
    size_t version;
    std::string str;
    size_t last = 0;
    f >> version;
    if(version > FVERSION)
    {
        mutex.unlock();
        return false;
    }

    if(version == 1) // backward compatibility
        f >> tmp; // sigmoid type

    // clear
    for(size_t i = 0; i < network.size(); ++i)
    for(size_t j = 0; j < network[i].size(); ++j)
        if(network[i][j])
            delete network[i][j];
    network.clear();
    in.clear();
    out.clear();

    f >> tmp;
    for(size_t i = 0; i < tmp; ++i)
        network.push_back(std::vector<Neuron*>());
    if(network.size() < 2)
    {
        mutex.unlock();
        return false;
    }

    f >> tmp;
    for(size_t i = 0; i < tmp; ++i)
        in.push_back(0);

    for(auto& i : network)
    {
        f >> tmp;
        for(size_t j = 0; j < tmp; ++j)
            i.push_back(new Neuron());

        for(auto& j : i)
        {
            f >> tmp;
            list[tmp] = j;
            if(version >= 4)
            {
                f >> str; // bias
                j->setBias(textToDouble(str));
            }

            size_t input_count;
            f >> input_count;
            for(size_t k = 0; k < input_count; ++k)
            {
                f >> tmp;
                if(tmp == 1)
                {
                    f >> tmp >> str; // str is the weight
                    j->addInput(list[tmp], textToDouble(str));
                }
                else
                {
                    j->addInput(&in[last]);
                    ++last;
                }
            }
        }
    }

    out.clear();
    for(size_t i = 0; i < network.back().size(); ++i)
        out.push_back(0);
    train_percent = 0;
    train_epoch = 0;
    training = false;
    //std::cout << "Loaded from " << filename << std::endl;
    mutex.unlock();
    return true;
}

bool NeuralNetwork::save(const std::string &filename)
{
    mutex.lock();
    std::ofstream f(filename.c_str(), std::ios::out | std::ios::trunc);
    if(!f)
    {
        mutex.unlock();
        return false;
    }
    std::map<Neuron*, size_t> list;
    size_t last = 0;
    f << (size_t)FVERSION << " " << network.size() << " " << in.size() << " ";
    for(auto& i : network)
    {
        f << i.size() << " ";
        for(auto& j : i)
        {
            if(list.find(j) == list.end())
            {
                list[j] = last;
                ++last;
            }
            f << list[j] << " " << doubleToText(j->getBias()) << " ";
            std::vector<NeuInput>& inputs = j->getInputs();
            f << inputs.size() << " ";
            for(auto& k : inputs)
            {
                if(k.isNeuron)
                {
                    f << (size_t)1 << " " << list[(Neuron*)k.ptr] << " " << doubleToText(k.weight) << " ";
                }
                else
                {
                    f << (size_t)0 << " ";
                }
            }
        }
    }
    //std::cout << "Saved to " << filename << std::endl;
    mutex.unlock();
    return true;
}

std::vector<std::vector<Neuron*> > NeuralNetwork::stealNetwork()
{
    std::vector<std::vector<Neuron*> > net = network;
    network.clear();
    in.clear();
    out.clear();
    return net;
}

void NeuralNetwork::setNetwork(std::vector<std::vector<Neuron*> > &net)
{
    clear();
    network = net;
    if(network.size() < 2)
        return;
    for(size_t i = 0; i < network[0].size(); ++i)
        in.push_back(0);
    for(size_t i = 0; i < network[0].size(); ++i)
        network[0][i]->addInput(&in[i]);
    for(size_t i = 0; i < network.back().size(); ++i)
        out.push_back(0);
}

NeuralNetwork* NeuralNetwork::merge(NeuralNetwork& A, NeuralNetwork& B)
{
    std::vector<std::vector<Neuron*> > netA = A.stealNetwork();
    std::vector<std::vector<Neuron*> > netB = B.stealNetwork();
    if(netB.size() > netA.size()) // network with the most layer is A
        netA.swap(netB);

    for(size_t i = 0; i < netB.size(); ++i)
    {
        std::vector<Neuron*>& ref = netA[i];
        if(i > 0)
            ref = netA[i+netA.size()-netB.size()];
        for(auto& j: netB[i])
            ref.push_back(j);
    }

    NeuralNetwork* R = new NeuralNetwork(netA);
    return R;
}

void NeuralNetwork::randomDropout()
{
    //std::cout << "Dropout (per layer): ";
    for(auto& i: network)
    {
        if(i.size() < 2)
            continue;
        std::uniform_int_distribution<> d1(1, i.size()/2);
        std::uniform_int_distribution<> d2(0, i.size()-1);
        int n = d1(gen);
        //std::cout << n << "/" << i.size() << " ";
        std::set<size_t> dropped;
        while(n > 0)
        {
            size_t j = d2(gen);
            if(dropped.find(j) == dropped.end())
            {
                i[j]->setDropout(true);
                dropped.insert(j);
                --n;
            }
        }
    }
    //std::cout << std::endl;
}

void NeuralNetwork::clearDropout()
{
    for(auto& i: network)
    {
        for(auto& j: i)
        {
            j->setDropout(false);
        }
    }
}
