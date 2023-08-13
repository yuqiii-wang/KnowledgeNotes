# Some Advanced JAVA Topics


## Annotation

Annotation is a function prior to being executed before a function.

It can help compiler perform checking and value initialization.

* All attributes of annotations are defined as methods, and default values can also be provided.

Here is an example, that `@Todo` inits some value before `incompleteMethod1` performs further business works.

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

### Common Annotations

* `@Override`

`@Override` tells the compiler that this method is an overridden method (metadata about the method), and if any such method does not exist in a parent class, then throw a compiler error (method does not override a method from its super class). 

```java
@Override
public String toString() {
   return "This is String Representation of current object.";
}
```

* `@Test`

The `@Test` annotation tells JUnit that the public void method to which it is attached can be run as a test case. 
To run the method, JUnit first constructs a fresh instance of the class then invokes the annotated method. 
Any exceptions thrown by the test will be reported by JUnit as a failure. 
If no exceptions are thrown, the test is assumed to have succeeded.

A simple test looks like this:
```java
public class Example {
  @Test
  public void method() {
     org.junit.Assert.assertTrue( new ArrayList().isEmpty() );
  }
}
```

## Interview Questions

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

* Package purposes

It only serves as a path by which a compiler can easily find the right definitions.

Namespace management

* Filename is often the contained class name

One filename should only have one class.

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

* Java Bean Concept

In computing based on the Java Platform, `JavaBeans` are classes that encapsulate many objects into a single object (the bean). 

The JavaBeans functionality is provided by a set of classes and interfaces in the java.beans package. Methods include info/description for this bean.

## Java NIO (Non Blocking IO) and EPoll

Traditionally, one thread manages one request/response.
This is wasteful since threads might get blocked by I/O operation.

Java NIO is the wrapper of Linux EPoll.

```java
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

public class NioServer {


    public static void main(String[] args) throws IOException {
      
        // NIO serverSocketChannel
        ServerSocketChannel serverSocketChannel = ServerSocketChannel.open();
        serverSocketChannel.bind(new InetSocketAddress(19002));

        // set non-blocking mode
        serverSocketChannel.configureBlocking(false);

        // launch epoll
        Selector selector = Selector.open();
        serverSocketChannel.register(selector, SelectionKey.OP_ACCEPT);

        while (true) {
            selector.select();
            Set<SelectionKey> selectionKeys = selector.selectedKeys();
            Iterator<SelectionKey> selectionKeyIterator = selectionKeys.iterator();
            while (selectionKeyIterator.hasNext()) {
                SelectionKey selectionKey = selectionKeyIterator.next();
                // onConnect
                if (selectionKey.isAcceptable()) {
                    ServerSocketChannel serverSocket= (ServerSocketChannel) selectionKey.channel();
                    SocketChannel socketChannel=serverSocket.accept();
                    socketChannel.configureBlocking(false);
                    socketChannel.register(selector,SelectionKey.OP_READ);
                    System.out.println("Connection established.");
                } else if (selectionKey.isReadable()) {
                    // onMessage
                    SocketChannel socketChannel= (SocketChannel) selectionKey.channel();
                    ByteBuffer byteBuffer=ByteBuffer.allocate(128);
                    int len=socketChannel.read(byteBuffer);
                    if (len>0){
                        System.out.println("Msg from client: " + new String(byteBuffer.array()));
                    }else if (len==-1){
                        System.out.println("Client disconnected: " + socketChannel.isConnected());
                        socketChannel.close();
                    }
               }
                selectionKeyIterator.remove();
            }
        }
    }
}
```

## Generics

Similar to template in cpp

```java
 public class GenericMethodTest {
   // generic method printArray
   public static < E > void printArray( E[] inputArray ) {
      // Display array elements
      for(E element : inputArray) {
         System.out.printf("%s ", element);
      }
      System.out.println();
   }

   public static void main(String args[]) {
      // Create arrays of Integer, Double and Character
      Integer[] intArray = { 1, 2, 3, 4, 5 };
      Double[] doubleArray = { 1.1, 2.2, 3.3, 4.4 };
      Character[] charArray = { 'H', 'E', 'L', 'L', 'O' };

      System.out.println("Array integerArray contains:");
      printArray(intArray);   // pass an Integer array

      System.out.println("\nArray doubleArray contains:");
      printArray(doubleArray);   // pass a Double array

      System.out.println("\nArray characterArray contains:");
      printArray(charArray);   // pass a Character array
   }
}
```

```java
public class MaximumTest {
   // determines the largest of three Comparable objects
   
   public static <T extends Comparable<T>> T maximum(T x, T y, T z) {
      T max = x;   // assume x is initially the largest
      
      if(y.compareTo(max) > 0) {
         max = y;   // y is the largest so far
      }
      
      if(z.compareTo(max) > 0) {
         max = z;   // z is the largest now                 
      }
      return max;   // returns the largest object   
   }
   
   public static void main(String args[]) {
      System.out.printf("Max of %d, %d and %d is %d\n\n", 
         3, 4, 5, maximum( 3, 4, 5 ));

      System.out.printf("Max of %.1f,%.1f and %.1f is %.1f\n\n",
         6.6, 8.8, 7.7, maximum( 6.6, 8.8, 7.7 ));

      System.out.printf("Max of %s, %s and %s is %s\n","pear",
         "apple", "orange", maximum("pear", "apple", "orange"));
   }
}
```

## Aspect-Oriented Programming (AOP)