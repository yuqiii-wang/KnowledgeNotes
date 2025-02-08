#include "CustomVector.hpp"

int main() {
    CustomVector<Person> vec;

    vec.emplace_back("Yuqi", 29, 'M') ;
    vec.emplace_back("Sexy Yuqi", 28, 'M') ;
    vec.emplace_back("Wild Yuqi", 27, 'M') ;
    vec.emplace_back("Crazy Yuqi", 26, 'M') ;
    vec.emplace_back("Magnificent Yuqi", 25, 'M') ;

    for (int idx = 0; idx < vec.get_size(); idx++) {
        std::cout << "Person: {" << vec[idx] << "}" << std::endl;
    }
    
    return 0;
}