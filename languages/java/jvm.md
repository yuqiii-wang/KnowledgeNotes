# Java Virtual Machine

JVM is a virtual machine, an abstract computer that has its own ISA, own memory, stack, heap, etc. 
It runs on the host OS and places its demands for resources to it.

It is a specification where diff companies have their own implementations, e.g. Oracle version, IBM version.


<div style="display: flex; justify-content: center;">
      <img src="imgs/jvm_arch.png" width="40%" height="30%" alt="jvm_arch" />
</div>
</br>


## Class Loading and Linking 

Loading means reading `.class` to memory.

The name must be unique `package` + `class`.
Java creates `java.lang.Class` to represent this class.

Java performs checking on the loaded `.class` such as file format, etc.

Allocate memory to `static`/global and set value to zeros.

## JVM vs JIT

Java Virtual Machine (JVM) is used in the java runtime environment(JRE). 
The original JVM was conceived as a bytecode interpreter.

Java Just In Time (JIT) compiler takes to be executed byte code and compiles it into machine code at run time.

<div style="display: flex; justify-content: center;">
      <img src="imgs/java_jit_vs_jvm.png" width="40%" height="30%" alt="java_jit_vs_jvm" />
</div>
</br>

## JVM Virtual Memory Layout

* Heap

* Method Area

* Program Counter Register

* JVM Stacks

### Common `OutOfMemoryError`

## JVM Garbage Collection (GC) Tuning

Resources need to be recycled when 
* reference count of an object is zero
* reachability analysis that if an object is not linked/used/owned by any other object

### ZGC (The Z Garbage Collector)

## `jvm` annotations
