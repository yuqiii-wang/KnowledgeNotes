# Smart Pointer's Realization

## Shared Pointer

Shared pointers track the shared ownership through a reference count property. When a pointer goes out of scope, it decreases one reference count from its total number of registered ownership. When it goes to zero, the pointed memory is freed.

```cpp
#include <iostream> //main header
#include <memory>   //for smart pointers
using namespace std;//namespace

int main()
{
    std::shared_ptr<int> sh1 (new int);   
    std::shared_ptr<int> sh2 = shp1;    
    auto sh3 = sh1;                       
    auto sh4 = std::make_shared<int>();

    cout << sh1.get() << endl;
    cout << sh2.get() << endl;
    cout << sh3.get() << endl;
    cout << sh4.get() << endl;

/* 
    output: only sh4 is new
    0x1c66c20
    0x1c66c20
    0x1c66c20
    0x1c66c70
*/

  return 0;  
}
```

### Shared Pointer counter

Depending on materialization, for `boost::shared_ptr`, a counter is defined in `private` in a `shared_ptr` container, in which it `new`s a counter. As a result, a `shared_ptr` counter resides in heap.

### Shared Pointer Passing Cost

When heavily passing shared pointer pointed var, use reference `const shared_ptr<T const>&` rather than by value (it creates a new pointer every time calling copy constructor `shared_ptr<T>::shared_ptr(const shared_ptr<T> &)`)
```cpp
void f(const shared_ptr<T const>& t) {...} 
```

## Weak Pointer

Dangling pointers and wild pointers are pointers that do not point to a valid object of the appropriate type. Weak pointers are used to "try" access the pointer to see if it is a dangling pointer by `lock()`.

Weak pointer manages more reference count than shared pointer, as it needs to track 

```cpp
// empty definition
std::shared_ptr<int> sptr;

// takes ownership of pointer
sptr.reset(new int);
*sptr = 10;

// get pointer to data without taking ownership
std::weak_ptr<int> weak1 = sptr;

// deletes managed object, acquires new pointer
sptr.reset(new int);
*sptr = 5;

// get pointer to new data without taking ownership
std::weak_ptr<int> weak2 = sptr;

// weak1 is expired!
if(auto tmp = weak1.lock())
    std::cout << "weak1 value is " << *tmp << '\n';
else
    std::cout << "weak1 is expired\n";

// weak2 points to new data (5)
if(auto tmp = weak2.lock())
    std::cout << "weak2 value is " << *tmp << '\n';
else
    std::cout << "weak2 is expired\n";
```
that outputs
```bash
weak1 is expired
weak2 value is 5
```

## Prefer `std::make_unique` and `std::make_shared` to direct use of `new`

Rule of thumb: Try to use standard tools as much as possible, otherwise, you risk program failure when OS or compiler upgrade to a higher version.

Smart pointer make is a simple forward operation as below:
```cpp
template<typename T, typename... Ts>
std::unique_ptr<T> make_unique(Ts&&... params)
{
    return std::unique_ptr<T>(new T(std::forward<Ts>(params)...));
}
```

## Circular Dependency Issues with `std::shared_ptr`, and `std::weak_ptr`

In the code below, by `p1->m_partner = p2;		p2->m_partner = p1;`, two shared pointers are dependent on each other. After `partnerUp()` is called, there are two shared pointers pointing to “Ricky” (ricky, and Lucy’s m_partner) and two shared pointers pointing to “Lucy” (lucy, and Ricky’s m_partner).

At the end of `main()`, the ricky's shared pointer goes out of scope first. However, it finds that its pointer has dependency on lucy's. To avoid dangling pointer issues, it does not deallocate lucy's pointer, and vice versa, lucy does not release ricky's pointer.

As a result, only `Person` constructor's `std::cout << m_name << " created\n";` got invoked, and they are not destroyed.

```cpp
class Person{
	std::string m_name;
	std::shared_ptr<Person> m_partner; // initially created empty

    Person(const std::string &name): m_name(name)	{
		std::cout << m_name << " created\n";
	}
	~Person()	{
		std::cout << m_name << " destroyed\n";
	}

    friend bool partnerUp(std::shared_ptr<Person> &p1, std::shared_ptr<Person> &p2)	{
		if (!p1 || !p2)
			return false;

		p1->m_partner = p2;
		p2->m_partner = p1;

		std::cout << p1->m_name << " is now partnered with " << p2->m_name << '\n';

		return true;
	}
}

int main(){
	auto lucy { std::make_shared<Person>("Lucy") }; // create a Person named "Lucy"
	auto ricky { std::make_shared<Person>("Ricky") }; // create a Person named "Ricky"

	partnerUp(lucy, ricky); // Make "Lucy" point to "Ricky" and vice-versa

	return 0;
}
```

### `std::weak_ptr` as the Solution to Shared Pointer Circular Dependency

A `std::weak_ptr` is an observer -- it can observe and access the same object as a `std::shared_ptr` but it is not considered an owner.

Given the same code above, just need to replace `std::shared_ptr<Person> m_partner;` with `std::weak_ptr<Person> m_partner;`. Functionally speaking, when ricky goes out of scope, a `std::weak_ptr` performs a double check that there are no other `std::shared_ptr` pointing at “Ricky” (the `std::weak_ptr` from “Lucy” doesn’t count). Therefore, it will deallocate “Ricky”. The same occurs for lucy.

However, `std::weak_ptr` are not directly usable (they have no `operator->`). To use a `std::weak_ptr`, you must first convert it into a `std::shared_ptr` (in practice, implicit conversion ).

## Auto Pointer and Unique Pointer

Note: `std::auto_ptr` is deprecated and `std::unique_ptr` is its replacement.

Same as `std::shared_ptr` but without counter.