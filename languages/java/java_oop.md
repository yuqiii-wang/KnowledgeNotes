# Java OOP (Object Oriented Programming)

## Inheritance
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

The nearest reached functions are called when there are multiple functions with same signatures (same name and arguments).

`super` can directlly call parent's methods.
```java
public class Animal {
   Animal(){}
   public void eat(){}
}

public class Dog extends Animal {
   super(); // calls Animal's constructor
   super.eat(); // calls Animal's eat method
}
```

## Polymorphism

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

* Overriding

Member functions of subclass with same names as that of a super class, these member functions override that of the super class.

## Abstraction

Object-oriented programming, abstraction is a process of hiding the implementation details from the user, only the functionality will be provided to the user. 
A class which contains the abstract keyword in its declaration is known as `abstract` class. Implementation of abstract class can be done via inheritance. This draws similarity with `virtual` declaration in cpp.

```java
// Employee is now abstruct and actual methods inside Employee can be coded in Salary
public abstract class Employee {}
public class Salary extends Employee {}
```

* Interfaces

A class implements an `interface`, thereby inheriting the abstract methods of the interface.  
It is a collection of abstract methods. It defines `abstract` type

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


## Encapsulation

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
