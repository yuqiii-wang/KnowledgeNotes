#include <iostream>
#include <ostream>

struct Person {
    Person() {
        std::cout << "Default person constructor is invoked." << std::endl;
    };

    Person(std::string name, int age, char gender):
    _name(name), _age(age), _gender(gender) {
        std::cout << "Person is identified as " << _name << std::endl;
    }

    // Move constructor
    Person(Person&& another_person) noexcept : 
    _name(another_person._name), _age(another_person._age), _gender(another_person._gender) {
        std::cout << "Move constructor is invoked." << std::endl;
    }

    // Copy constructor
    Person(const Person& another_person) : 
    _name(another_person._name), _age(another_person._age), _gender(another_person._gender) {
        std::cout << "Copy constructor is invoked." << std::endl;
    }

    friend std::ostream& operator<<(std::ostream& os, const Person& person);

protected:
    std::string _name;
    int _age;
    char _gender;
};