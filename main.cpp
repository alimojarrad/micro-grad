#include <iostream>
#include <vector>
#include <memory>
#include "neuron.h"  // Make sure to include your neuron and value headers

using namespace std;

int main() {
    // Create a neuron with 3 inputs
    neuron n(3, "tanh");

    // Print initial weights and bias
    cout << "Weights:" << endl;
    // Create input values as shared_ptr<value>
    vector<shared_ptr<value>> arr = {
        make_shared<value>(1.0),
        make_shared<value>(-2.0),
        make_shared<value>(0.5)
    };
    // Forward pass through the neuron
    auto out = n.forward(arr);
    for(auto i : n.getParameters()) {
        cout << i << endl;
    }
    // Print the raw output (before activation)
    cout << "Output of neuron: " << out->item() << endl;

    return 0;
}