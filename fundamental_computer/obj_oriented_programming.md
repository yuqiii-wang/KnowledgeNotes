# Object Oriented Programming

## Encapsulation

Encapsulation is a design pattern in which data are visible only to semantically related functions, and leads to frequent data hiding.

One typical employment is use of `set`/`get` method.

```cpp
class A {
private:
    int a;
public:
    A(){};
    ~A(){};
    void set_a(int in_a) {this->a = in_a;}
    int get_a() {return this->a;}
};
```

## Inheritance

Inheritance refers to derived classes inheriting base class member attrs and methods, as well as providing override services.

```cpp
class A {
public:
    A(){};
    virtual ~A(){};
    virtual int aMethod() {return 0;}
};      // Base class

class B : public A {
public:
    B(){};
    virtual ~B(){};
    int aMethod() override {return 1;}
};   // B derived from A
class C : public B{
    C(){};
    ~C(){};
};   // C derived from B
```

## Polymorphism

Polymorphism is the provision of a single interface to entities of different types (overload, compile time polymorphism) or or the use of a single symbol to represent multiple different types (override, run time polymorphism).

One typical use case is operator `+` is used with different behaviors (overload).

```cpp
int a = 1;
int c = 1 + a;  // + as arithmetic operator

std::string str = "1" + "+" + "1"; // + as string concatenation
```

Another one is by inheritance with common attributes of a base class
```cpp
class Animal {};
class Dog : public Animal {};
class Cat : public Animal {};
```

## Abstraction

Abstraction is about only exposing context-awared APIs to users. It benefits avoiding using duplicate code, preventing displaying low level high security process logic to end users.

One typical use case is math calculation, where a math lib only needs to expose an API while hiding complex computations from users.

Below shows that an end user only needs to call `multiplyMatrix(A, B)` to get the result.
```cpp
std::vector<std::vector<int>> multiplyPiecewise(std::vector<std::vector<int>>& A, std::vector<std::vector<int>>& B) {

    assert A.size() == B.size();
    std::vector<std::vector<int>> C(A.size(), A[0].size());

    for (int row_idx; row_idx < A.size(); row_idx++) {
        for (int col_idx; col_idx < A.size(); col_idx++) {
            C[row_indx][col_idx] = A[row_idx][col_idx] * B[row_idx][col_idx];
        }
    }
    return C;
}

std::vector<std::vector<int>> transposeMatrix(std::vector<std::vector<int>>& A){

    std::vector<std::vector<int>> C(A.size(), A[0].size());

    for (int row_idx; row_idx < A.size(); row_idx++) {
        for (int col_idx; col_idx < A.size(); col_idx++) {
            C[col_indx][row_idx] = A[row_idx][col_idx];
        }
    }
    return C;
}

std::vector<std::vector<int>> multiplyMatrix(std::vector<std::vector<int>>& A, std::vector<std::vector<int>>& B){
    return multiplyPiecewise(A, transposeMatrix(B));
}

// user only needs to call this to get matrixx multiplication result, do not need to concern nitty-gritty computation details
std::vector<std::vector<int>> C = multiplyMatrix(A, B); 
```