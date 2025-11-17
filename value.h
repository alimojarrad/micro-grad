#ifndef VALUE_H
#define VALUE_H
#include <iostream>
#include <list>
#include <functional>
#include <memory>
#include <vector>
using std::string;
using std::list;
using std::function;
using std::shared_ptr;
using std::vector;

struct value : public std::enable_shared_from_this<value>
{
private:
  string op;
  vector<shared_ptr<value>> _prev;
  function<void()> _backward = [](){} ;
public:
  shared_ptr<value> operator+(const shared_ptr<value>& v) const;
  shared_ptr<value> operator+(double v) const;
  shared_ptr<value> operator*(double v) const;
  shared_ptr<value> operator*(const shared_ptr<value>& v) const;
  shared_ptr<value> operator/(double v) const;
  shared_ptr<value> operator/(const shared_ptr<value>& v) const;
  shared_ptr<value> operator-(double v) const;
  shared_ptr<value> operator-(const shared_ptr<value>& v) const;
  shared_ptr<value> pow(double v) const;
  shared_ptr<value> tanh() const;
  vector<shared_ptr<value>> getParents();
  void backward();
  friend std::ostream& operator<<(std::ostream& os, const std::shared_ptr<value>& v);
  double data;
  double grad = 0.0;
  value(double data);
  double item();
};
shared_ptr<value> operator*(const shared_ptr<value>& left, double right);
shared_ptr<value> operator*(double left, const shared_ptr<value>& right);
shared_ptr<value> operator*(const shared_ptr<value>& left, const shared_ptr<value>& right);
shared_ptr<value> operator+(const shared_ptr<value>& left, double right);
shared_ptr<value> operator+(double left, const shared_ptr<value>& right);
shared_ptr<value> operator+(const shared_ptr<value>& left, const shared_ptr<value>& right);
shared_ptr<value> operator/(const shared_ptr<value>& left, double right);
shared_ptr<value> operator/(const shared_ptr<value>& left, const shared_ptr<value>& right);
shared_ptr<value> operator/(double left, const shared_ptr<value>& right);
shared_ptr<value> operator-(const shared_ptr<value>& left, double right);
shared_ptr<value> operator-(const shared_ptr<value>& left, const shared_ptr<value>& right);
shared_ptr<value> operator-(double left, const shared_ptr<value>& right);
#endif