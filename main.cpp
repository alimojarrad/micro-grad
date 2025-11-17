#include "value.h"
#include <iostream>

int main() {
    auto x = std::make_shared<value>(-1.5);
    auto y = std::make_shared<value>(2.0);
    
    auto a = x->tanh();      
    auto b = y->relu();      
    auto c = a + b;          
    auto d = c->sigmoid();   
    
    d->backward();
    
    std::cout << "=== Forward and Backward Pass ===\n";
    std::cout << "x: " << x << " (input to tanh)\n";
    std::cout << "y: " << y << " (input to relu)\n";
    std::cout << "a = tanh(x): " << a << "\n";
    std::cout << "b = relu(y): " << b << "\n";
    std::cout << "c = a + b: " << c << "\n";
    std::cout << "d = sigmoid(c): " << d << " <-- Output\n";
    
    return 0;
}
