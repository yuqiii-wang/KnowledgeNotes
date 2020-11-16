# Something about Design

There are Five creational design pattern:
1. Abstract factory groups object factories that have a common theme.
2. Builder constructs complex objects by separating construction and representation.
3. Factory method creates objects without specifying the exact class to create.
4. Prototype creates objects by cloning an existing object.
5. Singleton restricts object creation for a class to only one instance.

</br>

* Refactoring

Refactoring is a technique to improve the quality of existing code. It works by applying a series of small steps, each of which changes the internal structure of the code, while maintaining its external behavior. You begin with a program that runs correctly, but is not well structured, refactoring improves its structure, making it easier to maintain and extend.

* Factory Mode

In class-based programming, the factory method pattern is a creational pattern that uses factory methods to deal with the problem of creating objects without having to specify the exact class of the object that will be created. 

* Singleton

Singleton Pattern says that just"define a class that has only one instance and provides a global point of access to it".
In other words, a class must ensure that only single instance should be created and single object can be used by all other classes.

* Prototype

Prototype Pattern says that cloning of an existing object instead of creating new one and can also be customized as per the requirement.

* Observer

An Observer Pattern says that "just define a one-to-one dependency so that when one object changes state, all its dependents are notified and updated automatically". This draws similarity with broadcaster-to-listener mechanism.

* Chain of Responsibility

In chain of responsibility, sender sends a request to a chain of objects. The request can be handled by any object in the chain.
A Chain of Responsibility Pattern says that just "avoid coupling the sender of a request to its receiver by giving multiple objects a chance to handle the request". For example, an ATM uses the Chain of Responsibility design pattern in money giving process.