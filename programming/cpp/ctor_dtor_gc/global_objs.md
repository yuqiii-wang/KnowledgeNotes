# Global Objects

## Global Object Storage

Global objects are stored in ELF's static section, unlike local vars are stored in stack/heap areas.

## Init Globals

All global C++ variables that have constructors must have their constructor called before `main()`. The compiler builds a table of global constructor addresses that must be called, in order, before `main()` in a section called `.init_array`. 

The order that the compiler initializes global objects varies (depends on compiler)

### `.init_array`

|constructor addrs|
|-|
|global_constructor_1|
|global_constructor_2|
|global_constructor_3|
|...|

### Example

```cpp
class MyClass {
public:
    MyClass();
    virtual ~MyClass();

    void subscriptionHandler(const char *eventName, const char *data);
};

MyClass::MyClass() {
    // This is generally a bad idea. You should avoid doing this from a constructor.
    AnotherObj.subscribe("myEvent", &MyClass::subscriptionHandler, this);
}

MyClass::~MyClass() {

}

void MyClass::subscriptionHandler(const char *eventName, const char *data) {
    Log.info("eventName=%s data=%s", eventName, data);
}

// In this example, MyClass is a globally constructed object.
MyClass myClass;
```

### `__do_global_dtors_aux` and `__do_global_ctors_aux`

The addresses of constructors and destructors of static objects are each stored in a different section in ELF executable. for the constructors there is a section called *.CTORS* and for the destructors there is the *.DTORS* section.

the compiler creates two auxillary functions `__do_global_ctors_aux` and `__do_global_dtors_aux` for calling the constructors and destructors of these static objects, respectively.