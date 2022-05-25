# Personal Encountered Difficult Practices

## Naming 

Coz one vendor c++ code is not open source, when using Mongo DB, both invoke SSL modules and had problems. The two SSL drivers have their own implementations and misuse SSL drivers.

Solutions

Check compiled ASM code and found that there were naming conflicts

1. Separate the two modules and use MQ to connect
2. Change function names and recompile the code
3. Use diff namespace

## Template Deduction

Be careful it is passing by value or by reference when using complex templates.

## Architecture

Multi-inheritance management.

## GC

The returned value `default_value` derives from a temp var `""` that runs out of lifecycle when `m.FindOrDefault("key", "");` finishes, 
and any further use of `const std::string& a` raises a runtime error. 

```cpp
class MyStringMap {
public:
  const std::string& FindOrDefault(
            const std::string& key, const std::string& default_value) {
    const auto it = m_.find(key);
    if (it == m_.end()) return default_value;
    return it->second;
  }
private:
  std::map<std::string, std::string> m_;
};

const std::string& a = m.FindOrDefault("key", "");
```