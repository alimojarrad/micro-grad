#include "neuron.h"
#include <vector>
#include <memory>
#include <iostream>
#include <random>
using std::vector;
using std::shared_ptr;
using std::string;
using std::make_shared;
double random_uniform(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    return dis(gen);
}
neuron::neuron(int nin, string activation) {
    this->nin = nin;
    this->activation = activation;
    for(int i = 0; i < nin; i++) {
        this->weight.push_back(make_shared<value>(random_uniform(-0.5, 0.5)));
    }
    this->bias = make_shared<value>(random_uniform(-0.5, 0.5));
}
shared_ptr<value> neuron::forward(vector<shared_ptr<value>> x) {
    shared_ptr<value> output = this->bias;
    for(int i = 0; i < x.size(); i++) {
        output = ((this->weight[i]) * x[i]) + output;
    }
    if(this->activation == "relu") {
        return output->relu();
    } else if(this->activation == "sigmoid") {
        return output->sigmoid();
    } else if(this->activation == "tanh") {
        return output->tanh();
    } else {
        return output;
    }
}
vector<shared_ptr<value>> neuron::getParameters() const  {
    vector<shared_ptr<value>> params = weight;  // copy weights
    params.push_back(bias);                    // add bias
    return params;   
}