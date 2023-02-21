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

## Preprocessor Tag and Memory Alignment

`struct A` has possible memory invalid read/write on different compilation option of `DEBUG` for memory misalignment.

Such `struct` definition should be avoided.
```cpp
struct A
{
  int aVal;
#ifdef DEBUG
  int debug_aVal;
#endif
} a;
```

### Solution

Often sdk does not provide source code, and you need to use Valgrind and gdb to locate wronged code/memory leaked object.

## Program Killed by `SIGKILL`

### Some other process executed a `kill -9 <your-pid>`

Use `auditctl` to check which process initiated `sigkill`
1. to check if `audit` is up and running:
`ps -elf | grep audit`
2. add a rule to monitor `sigkill` by
`auditctl -a exit,always -S kill`; besides, use `auditctl -a exit,always -S kill -F a0=<pid-hex-format>` to monitor a particular pid, where the `<pid>` is in hex rather than decimal
3. When the `<pid>` is killed, check the log
`tail /var/log/audit/audit.log -n 100`
4. search for the keyword `type=SYSCALL` and `syscall=62`, locate the `sigkill`-initiated pid and what command that pid uses. For example, `comm="kill" exe="/usr/bin/kill"`
5. Use `ps -elf | grep <pid>` to check detail of the pid; Check `/var/log/messages` or `/var/log/syslog` to see OS logs if the `sigkill` is initiated by OS's process

### Kernel OOM killer decided that your process consumed too many resources, and terminated it.
Check  `/var/log/messages` or `/var/log/syslog` to see OS logs

Inside possible infinite loop, add `usleep(100);` to make `std::cout` having time buffer to actually output something, otherwise, since infinite loop does not give time to `std::cout` to print anything. 