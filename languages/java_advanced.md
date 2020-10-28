# Some Advanced JAVA Topics

* Multithreading

A lifecycle of Java multithreading object shows as below.
![alt text](imgs/java_multithreading_lifecycle.png "java_multithreading_lifecycle")

1. As a first step, you need to implement a run() method provided by a Runnable interface. This method provides an entry point for the thread and you will put your complete business logic inside this method. 

2. As a second step, you will instantiate a Thread object using the following constructor.

3. Once a Thread object is created, you can start it by calling start() method, which executes a call to run( ) method.

```java
class RunnableDemo implements Runnable {
   private Thread t;
   private String threadName;
   
   RunnableDemo( String name) {
      threadName = name;
      System.out.println("Creating " +  threadName );
   }
   
   public void run() {
      System.out.println("Running " +  threadName );
      try {
         for(int i = 4; i > 0; i--) {
            System.out.println("Thread: " + threadName + ", " + i);
            // Let the thread sleep for a while.
            Thread.sleep(50);
         }
      } catch (InterruptedException e) {
         System.out.println("Thread " +  threadName + " interrupted.");
      }
      System.out.println("Thread " +  threadName + " exiting.");
   }
   
   public void start () {
      System.out.println("Starting " +  threadName );
      if (t == null) {
         t = new Thread (this, threadName);
         t.start ();
      }
   }
}

public class TestThread {

   public static void main(String args[]) {
      RunnableDemo R1 = new RunnableDemo( "Thread-1");
      R1.start();
      
      RunnableDemo R2 = new RunnableDemo( "Thread-2");
      R2.start();
   }   
}
```

* Java Lock

* Annotation

One word to explain annotation is metadata and are only metadata and do not contain any business logic. The usage of annotation is mainly about decoupling of different parts of a project and avoids xml configuration.

For example,

```java
@Override
public String toString() {
   return "This is String Representation of current object.";
}
```

@Override tells the compiler that this method is an overridden method (metadata about the method), and if any such method does not exist in a parent class, then throw a compiler error (method does not override a method from its super class). 

All attributes of annotations are defined as methods, and default values can also be provided.

Here is an example.

```java
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@interface Todo {
   public enum Priority {LOW, MEDIUM, HIGH}
   public enum Status {STARTED, NOT_STARTED}
   String author() default "Yash";
   Priority priority() default Priority.LOW;
   Status status() default Status.NOT_STARTED;
}

// Usage goes as below
@Todo(priority = Todo.Priority.MEDIUM, author = "author_name", status = Todo.Status.STARTED)
public void incompleteMethod1() {
   //Some business logic is written
   //But it’s not complete yet
}
```

* Question: Integer equal comparison

Explained: Integer objects with a value between -127 and 127 are cached and return same instance (same addr), others need additional instantiation hence having different addrs.
```java
class D {
   public static void main(String args[]) {
      Integer b2 = 1;
      Integer b3 = 1;
      // print True
      System.out.println(b2 == b3);

      b2 = 128;
      b3 = 128;
      // print False
      System.out.println(b2 == b3);
   }
}
```

P.S. in Java 1.6, Integer calls `valueOf` when assigning an integer.
```java
public static Integer valueOf(int i) {
   if(i >= -128 && i <= IntegerCache.high)
      return IntegerCache.cache[i + 128];
   else
      return new Integer(i);
}
```

* Package real purposes

It only serves as a path by which a compiler can easily find the right definitions.

Namespace management

* Force garbage collection

When a reference has no object to refer, this reference is retreated. Forced reference deallocation can be by NULL: `ref = null;`.

Objects instantiated by `new` are often by reference when passed, while primitive types, string and constants are passed by value.

* Filename is often the contained class name

One filename should only have one class.

* Variable Init

Local variables inside a method of a class are not init and programmer should manually init them; while class-level variables have default init values.

* Constructor has no return type

Given this consideration, `box()` is not a constructor.
```java
class box{
   public void box(){}
}
```

* static

When using `static` to modify a variable, it should go with `final` to make it a constant.

Avoid using `static` to make a variable global.

`static` loads when a java class loads.

* Type Casting

We cast the Dog type to the Animal type. Because Animal is the supertype of Dog, this casting is called **upcasting**.
Note that the actual object type does not change because of casting. The Dog object is still a Dog object. Only the reference type gets changed. 

Here `Animal` is `Dog`'s super class. When `anim.eat();`, it actually calls `dogg.eat()`.
```java
Dog dog = new Dog();
Animal anim = (Animal) dog;
anim.eat();
```

Here, we cast the Animal type to the Cat type. As Cat is subclass of Animal, this casting is called **downcasting**.
```java
Animal anim = new Cat();
Cat cat = (Cat) anim;
```

Usage of downward casting, since it is more frequently used than upward casting.

Here, in the `teach()` method, we check if there is an instance of a Dog object passed in, downcast it to the Dog type and invoke its specific method, `bark()`.

```java
public class AnimalTrainer {
    public void teach(Animal anim) {
        // do animal-things
        anim.move();
        anim.eat();
 
        // if there's a dog, tell it barks
        if (anim instanceof Dog) {
            Dog dog = (Dog) anim;
            dog.bark();
        }
    }
}
```

1. Casting does not change the actual object type. Only the reference type gets changed.
2. Upcasting is always safe and never fails.
3. Downcasting can risk throwing a ClassCastException, so the instanceof operator is used to check type before casting.

* java class

The `Object` class is the parent class of all the classes in java by default. It provides useful methods such as `toString()`. It is defined in `Java.lang.Object`.

* Inner Class

```java
public class C
{
   class D{ void f3(){} }
   
	D f4()
	{
		D d = new D();
		return d;
	}

	public static void main(String[] args)
	{
      // C must be instantiated before instantiate C.D
		C c = new C(); 
		C.D d = c.f4();
		d.f3();
		 // D d=new D();//error!
	}
}

// Multiple class inheritance example by inner class
public class S extends C.D {} 
```
* HashMap vs HashTable
There are several differences between HashMap and Hashtable in Java:

Hashtable is synchronized, whereas HashMap is not. This makes HashMap better for non-threaded applications, as unsynchronized Objects typically perform better than synchronized ones.

Hashtable does not allow null keys or values. HashMap allows one null key and any number of null values.

* Java Container