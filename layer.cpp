#include "layer.h"
#include <iostream>
#include <memory>
#include <vector>
#include "value.h"
#include "neuron.h"

using std::shared_ptr;
using std::string;
using std::vector;

layer::layer(int nin, int nout, string activation) {
    this->nin = nin;
    this->nout = nout;
    this->activation = activation;
    for(int i = 0; i < nout; i++) {
        this->neurons.push_back(make_shared<neuron>(nin, activation));
    }
}
vector<shared_ptr<value>> layer::forward(const vector<shared_ptr<value>>& x) {
    vector<shared_ptr<value>> out;
    for(auto& i : this->neurons) {
        out.push_back(i->forward(x));
    }
    return out; 
}
vector<vector<shared_ptr<value>>> layer::getParameters() {
    vector<vector<shared_ptr<value>>> params;
    for(auto& i : this->neurons) {
        params.push_back(i->getParameters());
    }
    return params;
}