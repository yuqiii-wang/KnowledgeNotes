#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <memory>

#include "Person.hpp"

// compile by `g++ move_perf_forwarding.cpp -std=c++20`

struct Person;

template <typename T>
class CustomVector {
public:
    // data_(std::make_unique<T[]>(1)) inits a T array of size 1, invoked once default constructor 
    CustomVector() : size_(0), capacity_(1), data_(std::make_unique<T[]>(1)) {}

    template <typename... Args>
    void emplace_back(Args&&... args) {
        if (size_ >= capacity_) {
            // Resize the storage if needed
            resize();
        }
        // Construct the element in place using placement new and perfect forwarding
        new (&data_[size_]) T(std::forward<Args>(args)...);
        ++size_;
    }

    void resize() {
        capacity_ *= 2;
        // called default constructors for mem allocation (hence the default constructor is empty of code execution)
        // it allocates a block of contiguous mem for the T array
        // then use placement new to "move construct" the exact values to the allocated mem
        auto newData = std::make_unique<T[]>(capacity_);
        for (size_t i = 0; i < size_; ++i) {
            new (&newData[i]) T(std::move(data_[i])); // Move constructor
            data_[i].~T(); // Explicitly destroy the old elements
        }
        data_ = std::move(newData);
        std::cout << "===============================================" << std::endl;
        std::cout << "Resize triggered, new capacity is " << capacity_ << std::endl;
        std::cout << "New data addr is" << std::endl;
        print_data_addr();
    }

    T& operator[](size_t index) {
        if (index >= size_) {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }

    void print_data_addr(){
        for (size_t i = 0; i < size_; ++i) {
            std::cout << &data_[i] << std::endl;
        }
    }

    int get_size() {
        return size_;
    }

    ~CustomVector() {
        for (size_t i = 0; i < size_; ++i) {
            data_[i].~T(); // Explicitly destroy elements
        }
    }

private:
    size_t size_;
    size_t capacity_;
    std::unique_ptr<T[]> data_;
};


// Overload the << operator for the Person class
std::ostream& operator<<(std::ostream& os, const Person& person) {
    os << "Name: " << person._name << ", Gender: " << person._gender << ", Age: " << person._age;
    return os;
}  