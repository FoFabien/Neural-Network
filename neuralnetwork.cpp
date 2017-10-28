#include "neuralnetwork.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <future>
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
    for(auto &i: sizes)
    {
        network.push_back(std::vector<Neuron*>());
        for(size_t j = 0; j < i; ++j)
            network.back().push_back(new Neuron());
    }
    for(size_t i = 0; i < network[0].size(); ++i)
        network[0][i]->addInput(NeuLink(NeuItem(nullptr, false), NeuItem(network[0][i], true)));
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
    out.clear();
    train_percent = 0;
    train_epoch = 0;
    training = false;
    train_paused = false;
    train_stop = false;
}

bool NeuralNetwork::connectNeurons(NeuCoor a, NeuCoor b)
{
    if(a.layer >= network.size() || a.neuron >= network[a.layer].size() || b.layer >= network.size() || b.neuron >= network[b.layer].size())
        return false;
    if(a.layer == b.layer && a.neuron == b.neuron)
        return false;
    if(!network[b.layer][b.neuron]->isOutputOf(network[a.layer][a.neuron]))
    {
        NeuLink link(NeuItem(network[a.layer][a.neuron], true), NeuItem(network[b.layer][b.neuron], true));
        network[b.layer][b.neuron]->addInput(link);
        network[a.layer][a.neuron]->addOutput(link);
    }

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

void NeuralNetwork::setInput(std::vector<double> &values)
{
    if(values.size() != network[0].size())
    {
        std::cout << "input size error" << std::endl;
        return;
    }
    std::vector<double>::iterator it = values.begin();
    for(size_t i = 0; i < values.size(); ++i)
    {
        NeuLink& link = network[0][i]->getInputLinks()[0];
        if(!link.input.isNeuron)
            link.input.ptr = (void*)(&*(it+i));
    }
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
            {
                NeuLink link(NeuItem(network[layer-1][i], true), NeuItem(network[layer].back(), true), d(gen));
                network[layer].back()->addInput(link);
                network[layer-1][i]->addOutput(link);
            }
            for(size_t i = 0; layer+1 < network.size() && i < network[layer+1].size(); ++i)
            {
                NeuLink link(NeuItem(network[layer].back(), true), NeuItem(network[layer+1][i], true), d(gen));
                network[layer+1][i]->addInput(link);
                network[layer].back()->addOutput(link);
            }
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
    if(out.size() != network.back().size())
        out.resize(network.back().size(), 0.f);
    if(network.size() >= 2)
    {
        for(size_t i = 0; i < out.size(); ++i)
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
    if(network[b.layer][b.neuron]->isOutputOf(network[a.layer][a.neuron]))
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
            std::vector<NeuLink> &inputs = network[i][j]->getInputLinks();
            for(auto& k: inputs)
            {
                k.data->weight = d(gen);
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
    train_stop = true;
}

size_t NeuralNetwork::trainingState() const
{
    if(train_stop)
        return 0;
    if(train_paused)
        return 2;
    return 1;
}

bool NeuralNetwork::runTraining(std::vector<std::vector<double> > &inputs, const std::vector<std::vector<double> > &outputs, const size_t &epochs, const size_t &batch_size, const double &learning_rate, const double& momentum, const double &weight_decay,const  bool &dropout,
                                const std::string& save_file)
{
    if(inputs.size() != outputs.size() || inputs.empty())
        return false;

    size_t step = epochs / 10;
    if(epochs < 10)
        step = epochs;

    train_percent = 0;
    training = true;
    train_stop = false;
    train_paused = false;

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

        if(!train(inputs, outputs, batch_size, learning_rate, momentum, weight_decay, dropout))
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
    training = false;
    return true;
}

bool NeuralNetwork::train(std::vector<std::vector<double> >& inputs, const std::vector<std::vector<double> >& outputs, const size_t &batch_size, const double &learning_rate, const double &momentum, const double &weight_decay, const bool &dropout)
{
    mutex.lock();
    if(inputs.size() != outputs.size() || inputs.empty())
    {
        std::cout << "dataset size error" << std::endl;
        mutex.unlock();
        return false;
    }

    int sleep_counter = 0;

    clearDropout(); // set dropout
    if(dropout)
        randomDropout();

    size_t i;
    for(i = 0; i < inputs.size(); ++i)
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
                        std::vector<NeuLink>& links = network[j][k]->getOutputLinks();
                        for(auto& l: links)
                            sum.push_back(l.data->weight * ((Neuron*)l.output.ptr)->getNodeDelta());
                        std::sort(sum.begin(), sum.end(), std::greater<double>());
                    }
                    node_delta = tmp * (1 - tmp) /*derivative*/ * DoubleSum(sum) /*sum of all weight*node_delta of outputs*/;
                }
                network[j][k]->doGradient(node_delta);
                network[j][k]->setNodeDelta(node_delta);
            }
        }
        if((batch_size > 0 && i % batch_size == 0) || i == inputs.size()-1)
            for(size_t j = 1; j < network.size(); ++j)
                for(size_t k = 0; k < network[j].size(); ++k)
                    network[j][k]->applyDelta(learning_rate, momentum, weight_decay);
        ++sleep_counter;
        if(train_sleep_counter > 1 && sleep_counter % train_sleep_counter == 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
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
            std::cout << "\t";
            if(network[i][j]->isInputNeuron())
                std::cout << "Input ";
            std::cout << "Neuron " << j << " (addr=" << (int)(network[i][j]) << ") (bias=" << network[i][j]->getBias() << "): ";
            if(!detail)
            {
                std::cout << network[i][j]->getInputLinks().size() << " input(s)" << std::endl;
                continue;
            }
            std::cout << std::endl;
            std::vector<NeuLink> &inputs = network[i][j]->getInputLinks();
            for(auto& k: inputs)
            {
                std::cout << "\t\tInput weight: " << k.data->weight << "\tto ";
                if(k.input.ptr == nullptr)
                {
                    std::cout << "Invalid Pointer (did you set the inputs?)" << std::endl;
                }
                else if(k.input.isNeuron)
                {
                    std::cout << "Neuron (addr=" << (int)k.input.ptr << ")" << std::endl;
                }
                else
                {
                    std::cout << "Value (addr=" << (int)k.input.ptr << ")=" << (*(double*)(k.input.ptr)) << std::endl;
                }
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

#define FVERSION 6
bool NeuralNetwork::load(const std::string &filename)
{
    mutex.lock();
    std::ifstream f(filename.c_str(), std::ios::in);
    if(!f)
    {
        mutex.unlock();
        std::cout << "can't open " << filename << std::endl;
        return false;
    }
    std::map<size_t, Neuron*> list;
    size_t tmp;
    size_t version;
    bool boolean;
    std::string str;
    size_t last = 0;
    f >> version;
    if(version > FVERSION)
    {
        mutex.unlock();
        std::cout << filename << " : bad version" << std::endl;
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
    //in.clear();
    out.clear();

    f >> tmp;
    if(f.eof())
    {
        mutex.unlock();
        std::cout << filename << " : corrupt file" << std::endl;
        return false;
    }
    for(size_t i = 0; i < tmp; ++i)
        network.push_back(std::vector<Neuron*>());
    if(network.size() < 2)
    {
        mutex.unlock();
        std::cout << filename << " : invalid network" << std::endl;
        return false;
    }

    if(version < 5)
    {
        f >> tmp;
        if(f.eof())
        {
            mutex.unlock();
            std::cout << filename << " : corrupt file" << std::endl;
            return false;
        }
    }

    for(auto& i : network)
    {
        f >> tmp;
        if(f.eof())
        {
            mutex.unlock();
            std::cout << filename << " : corrupt file" << std::endl;
            return false;
        }
        for(size_t j = 0; j < tmp; ++j)
            i.push_back(new Neuron());

        for(auto& j : i)
        {
            f >> tmp;
            list[tmp] = j;
            if(version < 6) // is input neuron or not
            {
                if(&i == &network[0]) // we assume only the first layer contains only input neurons for backward compatibility
                    j->setInputNeuron(true);
            }
            else
            {
                f >> boolean;
                j->setInputNeuron(boolean);
            }
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
                    NeuLink n(NeuItem(list[tmp], true), NeuItem(j, true), textToDouble(str));
                    j->addInput(n);
                    list[tmp]->addOutput(n);
                }
                else
                {
                    j->addInput(NeuLink(NeuItem(nullptr, false), NeuItem(j, true)));
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
    f << (size_t)FVERSION << " " << network.size() << " ";
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
            f << list[j] << " " << j->isInputNeuron() << " " << doubleToText(j->getBias()) << " ";
            std::vector<NeuLink>& inputs = j->getInputLinks();
            f << inputs.size() << " ";
            for(auto& k : inputs)
            {
                if(k.input.isNeuron)
                {
                    f << (size_t)1 << " " << list[(Neuron*)k.input.ptr] << " " << doubleToText(k.data->weight) << " ";
                }
                else
                {
                    f << (size_t)0 << " ";
                }
            }
        }
    }
    mutex.unlock();
    return true;
}

std::vector<std::vector<Neuron*> > NeuralNetwork::stealNetwork()
{
    std::vector<std::vector<Neuron*> > net = network;
    network.clear();
    out.clear();
    return net;
}

void NeuralNetwork::setNetwork(std::vector<std::vector<Neuron*> > &net)
{
    clear();
    network = net;
    if(network.size() < 2)
        return;
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
    for(auto& i: network)
    {
        if(i.size() < 2)
            continue;
        std::uniform_int_distribution<> d1(1, i.size()/2);
        std::uniform_int_distribution<> d2(0, i.size()-1);
        int n = d1(gen);
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
