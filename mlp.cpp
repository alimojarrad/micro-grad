#include "mlp.h"
#include <iostream>
#include <iostream>
#include <memory>
#include <vector>
#include "value.h"
#include "neuron.h"

using std::shared_ptr;
using std::string;
using std::vector;

mlp::mlp(int nin, const vector<int>& sizes,  string activation) {
    int input_size = nin;
    for (int size : sizes) {
        layers.push_back(make_shared<layer>(input_size, size, activation));
        input_size = size;
    }
}

vector<shared_ptr<value>> mlp::forward(const vector<shared_ptr<value>>& x) {
    vector<shared_ptr<value>> out = x;
    for (auto& l : layers) {
        out = l->forward(out);
    }
    return out;
}

vector<vector<vector<shared_ptr<value>>>> mlp::getParameters() {
    vector<vector<vector<shared_ptr<value>>>> params;
    for (auto& l : layers) {
        params.push_back(l->getParameters());
    }
    return params;
}

vector<vector<shared_ptr<value>>> mlp::forward(const vector<vector<shared_ptr<value>>>& batch) {
    vector<vector<shared_ptr<value>>> out;
    for (auto& sample : batch) {
        out.push_back(this->forward(sample));
    }
    return out;
}
