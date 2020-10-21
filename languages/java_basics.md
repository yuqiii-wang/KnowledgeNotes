# Java Basics

* Inheritance
```java
class Super {
    // code
}
class Sub extends Super {
    // code
}
```

The Superclass reference variable can hold the subclass object, but using that variable you can access only the members of the superclass, so to access the members of both classes it is recommended to always create reference variable to the subclass.

If a class is inheriting the properties of another class. And if the members of the superclass have the names same as the sub class, to differentiate these variables we use `super` keyword.

Generally, the `implements` keyword is used with classes to inherit the properties of an interface. Interfaces can never be extended by a class.

* Overriding

Member functions of subclass with same names as that of a super class, these member functions override that of the super class.

* Polymorphism

Polymorphism is the ability of an object to take on many forms. Any Java object that can pass more than one IS-A test is considered to be polymorphic. 

IS-A is a way of saying: This object is a type of that object, e.g., Animal is the superclass of Mammal class -> Mammal IS-A Animal.

```java
// All the reference variables d, a, v, o refer to the same Deer object in the heap.
Deer d = new Deer();
Animal a = d;
Vegetarian v = d;
Object o = d;
```

At compile time, compiler validates statements by the variable type, while at runtime, JVM runs statements by what `new` operator has constructed.

```java
// In the example below, e is checked at compile time by the Employee class, while JVM (at runtime) runs code inside Salary.

public class Employee {}
public class Salary extends Employee {}

public class VirtualDemo {
    public static void main(String [] args) {
        Salary s = new Salary();
        Employee e = new Salary();
    }
}
```

* Abstruction

Object-oriented programming, abstraction is a process of hiding the implementation details from the user, only the functionality will be provided to the user. A class which contains the abstract keyword in its declaration is known as `abstract` class. Implementation of abstruct class can be done via inheritance. This draws similarity with `virtual` declaration in cpp.

```java
// Employee is now abstruct and actual methods inside Employee can be coded in Salary
public abstract class Employee {}
public class Salary extends Employee {}
```

* Encapsulation

Encapsulation in Java is a mechanism of wrapping the data (variables) and code acting on the data (methods) together as a single unit. In encapsulation, the variables of a class will be hidden from other classes, and can be accessed only through the methods of their current class. 

There are two ideas to remember:

1. Declare the variables of a class as private.
2. Provide public setter and getter methods to modify and view the variables values.

```java
// use setXXX(newXXX); and getXXX() to encapsulate data
public class EncapTest {
   private int var;

    public int getVar() {
      return var;
   }

    public void setVar( int newVar) {
      var = newVar;
   }
}
```

* Interfaces

A class implements an `interface`, thereby inheriting the abstract methods of the interface.  It is a collection of abstract methods.

```java
interface Animal {
   public void eat();
   public void travel();
}

interface Creature {}

public class MammalInt implements Animal {

   public void eat() {
      System.out.println("Mammal eats");
   }

   public void travel() {
      System.out.println("Mammal travels");
   } 
}

// Inheritance can be done by 'extends'
public interface MammalInt extends Animal, Creature {}
```

* Generics

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

* Internet

Sockets provide the communication mechanism between two computers using TCP. A client program creates a socket on its end of the communication and attempts to connect that socket to a server.

When the connection is made, the server creates a socket object on its end of the communication.

```java
// A java client socket example

import java.net.*;
import java.io.*;

public class GreetingClient {

   public static void main(String [] args) {
      String serverName = args[0];
      int port = Integer.parseInt(args[1]);
      try {
         System.out.println("Connecting to " + serverName + " on port " + port);
         Socket client = new Socket(serverName, port);
         
         System.out.println("Just connected to " + client.getRemoteSocketAddress());
         OutputStream outToServer = client.getOutputStream();
         DataOutputStream out = new DataOutputStream(outToServer);
         
         out.writeUTF("Hello from " + client.getLocalSocketAddress());
         InputStream inFromServer = client.getInputStream();
         DataInputStream in = new DataInputStream(inFromServer);
         
         System.out.println("Server says " + in.readUTF());
         client.close();
      } catch (IOException e) {
         e.printStackTrace();
      }
   }
}
```