#include "Person.hpp"

class Passenger : public Person {
public:
    Passenger(){};

    Passenger(std::string name, int age, char gender, uint64_t* luggage){
        _name = name;
        _age = age;
        _gender = gender;
    }
private:
    int _max_luggage_size = 1 << 16;
};