// struct A
// {
//     virtual A* get() const = 0;
// };

// template <typename T>
// struct B : virtual A
// {
//     T* get() const override {
//         return new T();
//     }
// };

template<typename T>
struct A
{
    virtual A* get() const = 0;
};

template <typename T>
struct B : virtual A<B<T>>
{
    T* get() const override {
        return new T{*static_cast<const T*>(this)};
    }
};

struct C : B<C>
{
    C(){};
};

int main()
{
    C c;
    return 0;
}