# Intrusive and non-intrusive containers

The main difference between intrusive containers and non-intrusive containers is that in C++ non-intrusive containers store **copies** of values passed by the user. 

For example, `myclass_list.push_back(myclass);` runs a copy constructor when attached a node `myclass` to the existing list.
```cpp
#include <list>
#include <assert.h>

int main()
{
   std::list<MyClass> myclass_list;

   MyClass myclass(...);
   myclass_list.push_back(myclass); // copy constructor

   //The stored object is different from the original object
   assert(&myclass != &myclass_list.front());
   return 0;
}
```

On the other hand, an intrusive container does not store copies of passed objects, but it stores the objects themselves.

For example, in `MyClass` should define `MyClass *next;` and `MyClass *previous;` that are used in intrusive list.
In other words, intrusive container takes a pointer of an object rather than running a copy constructor.
```cpp
class MyClass
{
   MyClass *next;
   MyClass *previous;
   //Other members...
};

int main()
{
   acme_intrusive_list<MyClass> list;

   MyClass myclass;
   list.push_back(myclass);

   //"myclass" object is stored in the list
   assert(&myclass == &list.front());
   return 0;
}
```