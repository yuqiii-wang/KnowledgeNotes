# Singleton

The Singleton pattern ensures that a class has only one instance and provides a global point of access to that instance.

```cpp
class Singleton
{
public:
    static Singleton &Instance()
    {
        // This line only runs once, thus creating the only instance in existence
        static std::auto_ptr<Singleton> instance( new Singleton );
        // always returns the same instance
        return *instance; 
    }
private:
    Singleton(){}
    ~Singleton(){}
};
```