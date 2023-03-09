/*
    Use template to get a private member of a class
*/

#include <iostream>

// A has a private member aVal
// `static` is required for this member to be passed such as `A::aVal`
class A {
private:
    static int aVal;
};
int A::aVal = 42;

// define a function that can access A.aVal
int* get();

// in compilation time to deduce and obtain A's private member addr
// for template only works in compilation time, only static private member can be deduced
// `friend` is used to include `int* get()` as its implementation, otherwise `int* get();` is undefined
template<int *x>
struct Get{
    friend int* get(){return x;}
};

// According to http://eel.is/c++draft/temp.spec#general-6,
// the template arguments of explicit instantiation declarations 
// are permitted to name “private types or objects that would normally not be accessible”,
// in other words, template does not do access scope check under this circumstance
template struct Get<&A::aVal>;

int main()
{
    std::cout << *get() << std::endl;

    return 0;
}
