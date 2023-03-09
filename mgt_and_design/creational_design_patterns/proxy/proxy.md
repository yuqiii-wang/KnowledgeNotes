# Proxy Design

In proxy pattern, a class (acted as a proxy) represents functionality of another class. 

In practice, it helps links multiple objects of different classes but should be arranged in the same business logic.

For example, a business logic requires arranging some seats for some people.
However, people are "virtual" that makes the universal arrangement impractical.
```cpp
// virtual base class
class Person {
public:
    virtual char getGender() = 0;
    virtual int getGrade() = 0;
    virtual std::string getMajor() = 0;
};

class Professor : public Person {...};
class Student : public Person {...};

// Person/base class is virtual, only the derived class object can be instantiated
Person classRoomSeats[100];
Professor p1;
Student s1;
classRoomSeats[seatIdx++] = p1;
classRoomSeats[seatIdx++] = s1;
```

Solution to the above problem can be using pointer and dynamic cast such as 
```c++
Person* classRoomSeats[100];
classRoomSeats[seatIdx++] = dynamic_cast<Person*>(new Professor());
classRoomSeats[seatIdx++] = dynamic_cast<Person*>(new Student());
```

There is another elegant alternative that encapsulates complicated pointer arrangement: define `class PersonSurrogate;`, then allocate seats by `PersonSurrogate classRoomSeats[100];`.
In other words, person proxy collectively arranges what people sit in `classRoomSeats[100]`, representing what `Person` should do.

## Proxy Implementation

First, define a `copy()` operator in classes that have business meaning.

```cpp
// virtual base class
class Person {
public:
    virtual char getGender() = 0;
    virtual int getGrade() = 0;
    virtual std::string getMajor() = 0;

    virtual Person* copy() = 0;
    virtual ~Person() {};
};

class Professor : public Person {
public:
    ...
    virtual Person* copy() {
        return new Professor(*this);
    }
};

class Student : public Person {
public:
    ...
    virtual Person* copy() {
        return new Student(*this);
    }
};
```

Then, define a proxy `class PersonSurrogate;` that represents what `Person` should do.
```cpp
class PersonSurrogate
{
public:
    PersonSurrogate(): p(nullptr) {}
    PersonSurrogate(const Person& _p): p(_p.copy()) {}
    ~PersonSurrogate(){ delete p; }
    PersonSurrogate(const PersonSurrogate& _p): p( _p.p ? _p.copy() : nullptr) {} // copy constructor: use already pointed object rather than copying it

    PersonSurrogate& operator=( const PersonSurrogate& _p) { // copy assignment constructor
        if (this != &_p) {
            delete p;
            p = _p.p ? _p.p->copy() : nullptr;
        }
        return *this;
    }
    
    ... // define for `getGender()`, `getGrade()` and `getMajor()`

private:
    Person* p;
}
```

To use it, just replace `Person` with `PersonSurrogate` such as
```cpp
PersonSurrogate classRoomSeats[100];
Professor p1;
Student s1;
classRoomSeats[seatIdx++] = p1;
classRoomSeats[seatIdx++] = s1;
```