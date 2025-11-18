#ifndef NEURON_H
#define NEURON_H
#include "value.h"
#include <vector>
#include <memory>
#include <iostream>
using std::vector;
using std::shared_ptr;
using std::string;
using std::make_shared;
class neuron {
    private:
      int nin;
      vector<shared_ptr<value>> weight;
      shared_ptr<value> bias;
      vector<shared_ptr<value>> parameters;
      string activation;
    public:
      vector<shared_ptr<value>> getParameters();
      shared_ptr<value> forward(vector<shared_ptr<value>> x);
      neuron(int nin, string activation);
};
#endif