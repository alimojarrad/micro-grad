#include "value.h"
#include <iostream>
#include <unordered_set>
#include <memory>
#include <cmath>
using namespace std;
value::value(double data) {
    this->data = data;
}
shared_ptr<value> value::operator+(double v) const {
    double out = data + v;
    auto res = make_shared<value>(out);
    res->_prev.push_back(shared_ptr<value>(const_cast<value*>(this), [](value*) {}));
    auto other = make_shared<value>(v);
    res->_prev.push_back(other);
    res->op = "+";
    auto self = shared_ptr<value>(const_cast<value*>(this), [](value*) {});
    res->_backward = [self, other, res]() {
    self->grad += 1 * res->grad;
    other->grad += 1 * res->grad;
    };
    return res;
}
shared_ptr<value> value::operator+(const shared_ptr<value>& v) const {
    double out = this->data + v->data;
    auto res = make_shared<value>(out);
    res->_prev.push_back(shared_ptr<value>(const_cast<value*>(this), [](value*) {}));
    res->_prev.push_back(v);
    res->op = "+";
    auto self = shared_ptr<value>(const_cast<value*>(this), [](value*) {});
    res->_backward = [self, res, v]() {
        self->grad += res->grad * 1;
        v->grad += self->grad * 1;
    };
    return res;
}
shared_ptr<value> value::operator*(double v) const {
    double out = v * this->data;
    auto res = make_shared<value>(out);
    res->_prev.push_back(shared_ptr<value>(const_cast<value*>(this), [](value*) {}));
    auto other = make_shared<value>(v);
    res->_prev.push_back(other);
    res->op = "*";
    auto self = shared_ptr<value>(const_cast<value*>(this), [](value*) {});
    res->_backward = [self, res, other]() {
        self->grad += res->grad * other->data;
        other->grad += self->data * res->grad;
    };
    return res;
}
shared_ptr<value> value::operator*(const shared_ptr<value>& v) const {
    double out = this->data * v->data;
    auto res = make_shared<value>(out);
    res->_prev.push_back(shared_ptr<value>(const_cast<value*>(this), [](value*) {}));
    res->_prev.push_back(v);
    res->op = "*";
    auto self = shared_ptr<value>(const_cast<value*>(this), [](value*) {});
    res->_backward = [self, res, v]() {
        self->grad += res->grad * v->data;
        v->grad += self->data * res->grad;
    };
    return res;
}
shared_ptr<value> value::operator/(double v) const {
    if(v != 0.0) {
        double out = this->data / v;
        auto res = make_shared<value>(out);
        res->_prev.push_back(shared_ptr<value>(const_cast<value*>(this), [](value*) {}));
        auto other = make_shared<value>(v);
        res->_prev.push_back(other);
        res->op = "/";
        auto self = shared_ptr<value>(const_cast<value*>(this), [](value*) {});
        res->_backward = [self, other, res]() {
        self->grad += (1.0 / other->data) * res->grad;
        other->grad += (-self->data / (other->data * other->data)) * res->grad;
    };
        return res;
    } else {
        throw runtime_error("cannot be divided by 0");
    }
}
shared_ptr<value> value::operator/(const shared_ptr<value>& v) const {
    if(v->data != 0.0) {
        double out = this->data / v->data;
        auto res = make_shared<value>(out);
        res->_prev.push_back(shared_ptr<value>(const_cast<value*>(this), [](value*) {}));
        res->_prev.push_back(v);
        res->op = "/";
        auto self = shared_ptr<value>(const_cast<value*>(this), [](value*) {});
        res->_backward = [self, v, res]() {
        self->grad += (1.0 / v->data) * res->grad;
        v->grad += (-self->data / (v->data * v->data)) * res->grad;
    };
        return res;
    } else {
        throw runtime_error("Cannot be divided by zero");
    }
}
shared_ptr<value> value::operator-(double v) const {
    double out = this->data - v;
    auto res = make_shared<value>(out);
    res->_prev.push_back(shared_ptr<value>(const_cast<value*>(this), [](value*) {}));
    auto other = make_shared<value>(v);
    res->_prev.push_back(other);
    res->op = "-";
    auto self = shared_ptr<value>(const_cast<value*>(this), [](value*) {});
    res->_backward = [self, other , res]() {
        self->grad += 1 * res->grad;
        other->grad += -1 * res->grad;
    };
    return res;
}
shared_ptr<value> value::operator-(const shared_ptr<value>& v) const {
    double out = this->data - v->data;
    auto res = make_shared<value>(out);
    res->_prev.push_back(shared_ptr<value>(const_cast<value*>(this), [](value*) {}));
    res->_prev.push_back(v);
    res->op = "-";
    auto self = shared_ptr<value>(const_cast<value*>(this), [](value*) {});
    res->_backward = [self, res, v]() {
        self->grad += 1 * res->grad;
        v->grad += -1 * res->grad;
    };
    return res;
}
shared_ptr<value> value::pow(double v) const {
    double out = std::pow(this->data , v);
    auto res = make_shared<value>(out);
    res->_prev.push_back(shared_ptr<value>(const_cast<value*>(this), [](value*) {}));
    auto other = make_shared<value>(v);
    res->_prev.push_back(other);
    res->op = "^";
    auto self = shared_ptr<value>(const_cast<value*>(this), [](value*) {});
    res->_backward = [self, res, v]() {
        self->grad += v * (std::pow(self->data , (v - 1))) * res->grad;
    };
    return res;
}
shared_ptr<value> value::tanh() const {
    auto val = std::tanh(this->data);
    auto res = make_shared<value>(val);
    res->_prev.push_back(shared_ptr<value>(const_cast<value*>(this), [](value*) {}));
    auto self = shared_ptr<value>(const_cast<value*>(this), [](value*) {});
    res->_backward = [self, res, val]() {
        self->grad += (1 - std::pow(val , 2)) * res->grad;
    };
    return res;
}
void value::backward() {
    vector<shared_ptr<value>> topo;
    unordered_set<value*> visited;
    function<void(shared_ptr<value>)> build = [&](shared_ptr<value> v) {
        if (!v || visited.count(v.get())) return;
        visited.insert(v.get());
        for (auto& prev : v->_prev) {
            build(prev);
        }
        topo.push_back(v);
    };

    build(shared_from_this());
    this->grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
    }
}
shared_ptr<value> operator*(const shared_ptr<value>& left, double right) {
    return (*left) * right; 
}

shared_ptr<value> operator*(double left, const shared_ptr<value>& right) {
    return (*right) * left;
}

shared_ptr<value> operator*(const shared_ptr<value>& left, const shared_ptr<value>& right) {
    return (*left) * right; 
}
shared_ptr<value> operator/(const shared_ptr<value>& left, double right) {
    return *left / right;
}
shared_ptr<value> operator/(const shared_ptr<value>& left, const shared_ptr<value>& right) {
    return (*left) / right;
}
shared_ptr<value> operator/(double left, const shared_ptr<value>& right) {
    return *make_shared<value>(left) / right;
}
shared_ptr<value> operator+(const shared_ptr<value>& left, double right) {
    return (*left) + right;
}
shared_ptr<value> operator+(double left, const shared_ptr<value>& right) {
    return *right + left;
}
shared_ptr<value> operator+(const shared_ptr<value>& left, const shared_ptr<value>& right) {
    return (*left) + right;
}
shared_ptr<value> operator-(const shared_ptr<value>& left, double right) {
    return *left - right;
}
shared_ptr<value> operator-(const shared_ptr<value>& left, const shared_ptr<value>& right) {
    return *left - right;
}
shared_ptr<value> operator-(double left, const shared_ptr<value>& right) {
    return -1*right + left;
}
std::ostream& operator<<(std::ostream& os, const shared_ptr<value>& v) {
    os << "Value(data=" << v->data << ", grad : " << v->grad << " )";
    return os;
}
double value::item() {
    return data;
}
vector<shared_ptr<value>> value::getParents() {
    return this->_prev;
}