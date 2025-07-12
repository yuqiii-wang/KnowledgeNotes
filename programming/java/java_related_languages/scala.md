# Scala

Scala can be considered as specialized Java dedicated for **data manipulation** with added **syntactic sugar** but no optimization on performance for Scala is compiled to Java bytecode that **runs on JVM**.

Scala object on standard collections are **not inherently faster** than a well-written Java Stream equivalent.
The primary benefits of Scala's functional collections are **developer productivity, code safety, and readability, not raw performance**.

Scala sees performance optimizations in **immutable** objects.
Scala vs Java have below data assumptions:

* Scala: "Things are assumed immutable unless declared mutable."
* Java (Traditional): "Things are assumed mutable unless declared immutable."

## Scala Basic Syntax

### Scala vs. Java: Syntax Comparison Table

| Feature / Concept | Scala Syntax & Explanation | Java (Modern) Syntax & Explanation / Equivalent |
| :--- | :--- | :--- |
| **Variable Declaration** | `val x = 10` (immutable, like `final`)<br>`var y = 20` (mutable)<br><br>Type inference is standard. `val` is strongly preferred. | `final int x = 10;`<br>`int y = 20;`<br><br>Type must be declared explicitly (`var` is available since Java 10 for local variables). |
| **Method / Function** | `def add(x: Int, y: Int): Int = x + y`<br><br>Uses `def`. Parameter types come after the name (`name: Type`). Return type is after the parameter list. The `=` separates signature from implementation. | `public int add(int x, int y) {`<br> `  return x + y;`<br>`}`<br><br>Return type comes first. Body is always in `{}`. |
| **Class & Constructor** | `class Person(val name: String, val age: Int) { ... }`<br><br>The primary constructor is part of the class signature. Declaring parameters with `val` or `var` makes them public fields automatically. Extremely concise. | `public class Person {`<br>`  private final String name;`<br>`  private final int age;`<br>`  public Person(String name, int age) {`<br>`    this.name = name;`<br>`    this.age = age;`<br>`  }`<br>`  // Getters...`<br>`}`<br><br>Requires boilerplate for constructor, fields, and getters. |
| **Data Class / POJO** | `case class Person(name: String, age: Int)`<br><br>A "super-class" that automatically provides an immutable constructor, getters, `equals()`, `hashCode()`, `toString()`, a `copy()` method, and pattern matching support. | `public record Person(String name, int age) { }`<br><br>Java 16+ `record` is the direct equivalent. It provides an immutable constructor, accessors, `equals()`, `hashCode()`, and `toString()`. For older Java, you'd write a full POJO or use a library like Lombok. |
| **Singleton / Static** | `object StringUtils { ... }`<br><br>An `object` is a singleton instance of a class. It's used for utility methods and constants, effectively replacing Java's `static` members. | `public final class StringUtils {`<br>`  private StringUtils() {} // Prevent instantiation`<br>`  public static String someMethod() { ... }`<br>`}`<br><br>Typically achieved with a class containing only `static` methods and a private constructor. |
| **Interfaces / Mixins** | `trait Speaker { ... }`<br><br>A `trait` is like a Java interface but can also contain fields and implemented methods. A class can extend one class but mix in multiple traits using `with`. | `public interface Speaker { ... }`<br><br>Java 8+ interfaces with `default` methods are very similar. The key difference is that traits can have state (instance variables), whereas interfaces cannot (only `static final` constants). |
| **Controlled Inheritance** | `sealed trait Shape`<br><br>When a `trait` or `class` is `sealed`, all its direct subclasses must be declared in the same file. This allows the compiler to check for exhaustiveness in pattern matching. | `public sealed interface Shape permits Circle, Rectangle { ... }`<br><br>Java 17+ `sealed` classes and interfaces are the direct equivalent, providing the same compile-time safety for `switch` expressions. |
| **Pattern Matching** | `shape match {`<br>`  case Circle(r) => 3.14 * r * r`<br>`  case Square(s) => s * s`<br>`}`<br><br>A powerful `switch` statement that can deconstruct objects (`unapply`), match by type, and include guards (`if` conditions). | `// Java 21+`<br>`return switch (shape) {`<br>`  case Circle(var r) -> Math.PI * r * r;`<br>`  case Square(var s) -> s * s;`<br>`  // ... other shapes`<br>`};`<br><br>Modern Java's "Pattern Matching for switch" is very similar. Older Java used a series of `if (instanceof ...)` checks. |
| **Null Safety** | `def findUser(id: Int): Option[User]`<br><br>Returns `Some(user)` or `None`. Forces the caller to handle the absence of a value, preventing `NullPointerException`s. | `public Optional<User> findUser(int id)`<br><br>Java's `Optional` class serves the exact same purpose, returning `Optional.of(user)` or `Optional.empty()`. |
| **String Formatting** | `val name = "Alice"`<br>`val msg = s"Hello, $name"`<br><br>String interpolation is built-in with the `s` prefix. More complex expressions can be used: `s"Age: ${user.age}"`. | `String name = "Alice";`<br>`String msg = String.format("Hello, %s", name);`<br>`// or`<br>`String msg = "Hello, " + name;`<br><br>Uses `String.format()` or simple concatenation. |
| **Collections** | `val list = List(1, 2, 3)`<br>`val newList = list.map(_ * 2)`<br><br>Collections are immutable by default. The collections library is extremely rich with functional operators (`map`, `filter`, `fold`, etc.). | `List<Integer> list = List.of(1, 2, 3); // immutable`<br>`List<Integer> newList = list.stream()`<br>`  .map(i -> i * 2)`<br>`  .collect(Collectors.toList());`<br><br>Java's Streams API provides similar functional operators. Default collections like `ArrayList` are mutable. |
| **Semicolons** | Optional. Only needed to separate multiple statements on the same line. | Required at the end of every statement. |

---

### Templating with `<:` nd `>:`

* Upper Bounds (`<:`): Constrain a type parameter to be a subtype of a specified type.

```scala
// A method that accepts any type T that is a subtype of Number
def addNumbers[T <: Number](a: T, b: T): Double = {
  a.doubleValue() + b.doubleValue()
}
```

* Lower Bounds (`>:`): Constrain a type parameter to be a supertype of a specified type.

```scala
// A method that prepends an element to a List, ensuring type compatibility.
def prepend[T](elem: T, list: List[T]): List[T] = elem :: list

val nums = prepend(1, List(2, 3, 4))
```

### Syntactic Sugar

#### The `apply` syntactic sugar of invocation (it is compiler-recognized)

```scala
myObject(arg1, arg2)
```

is a shortcut for

```scala
myObject.apply(arg1, arg2)
```

#### Iteration Mapping

```scala
val numbers = List(1, 2)
val letters = List('a', 'b')

val pairs = for {
  n <- numbers  // for each number 'n' in 'numbers'
  c <- letters  // and for each character 'c' in 'letters'
} yield (n, c) // create a tuple (n, c)

println(pairs)
```

is equivalent to 

```scala
val pairsDeSugared = numbers.flatMap { n =>
  letters.map { c =>
    (n, c)
  }
}

println(pairsDeSugared)
// Output: List((1,a), (1,b), (2,a), (2,b))
```

#### Iteration Variable Placeholder

```scala
val nums = List(1, 2, 3, 4)

// Find all even numbers
val evens = nums.filter(_ % 2 == 0) // _ represents each element
```

is equivalent to

```scala
val evensDeSugared = nums.filter(x => x % 2 == 0)
```

#### Update Mapping

```scala
import scala.collection.mutable

val userAges = mutable.Map("Alice" -> 30, "Bob" -> 40)

// Looks like built-in array/map assignment
userAges("Charlie") = 25
```

is equivaslent to

```scala
// The compiler transforms the line above into this:
userAges.update("Charlie", 25)
```

#### List Appending New Elements

```scala
val list1 = List(1, 2, 3)
val list2 = list1 :+ 4 // `:+` returns a NEW list

// list1 is still List(1, 2, 3)
// list2 is List(1, 2, 3, 4)
```

#### Assignment

```scala
// Example with a Tuple
def getUserInfo(): (String, Int) = {
  ("Alice", 30) // A function returning a tuple
}

// De-structure the tuple directly into two variables
val (name, age) = getUserInfo()
```

#### Collection Concatenation and Merge

The `++` operator can do concatenation (for list) and merge (for set/map ).

```scala
val set1 = Set(1,2,3)
val set2 = Set(3,4,5)
println(set1 ++ set2)  // Output: Set(1,2,3,4,5)
```

## Scala Low-Level Implementation and Performance Considerations

Scala code is **not inherently faster** than well-written Java Stream equivalent.

Scala code compiles down to JVM bytecode, just like Java code.
This means at the lowest level of execution, both languages are subject to the same Just-In-Time (JIT) compiler optimizations, garbage collection, and overall runtime characteristics of the Java Virtual Machine.

||Compilation|Performance|
|:---|:---|:---|
|`trait`|Compiles to a standard Java `interface`|Negligible Overhead. This is an extremely common pattern. The JIT compiler is excellent at inlining these static method calls|
|`case class`|A standard Java class where the compiler has automatically generated the bytecode for `equals()`, `hashCode()`, `toString()`, `copy()`,  `apply()`, `unapply()` methods|No direct overhead. It's just a regular class. The fact that it's immutable can be a design-level performance win in concurrent code (no need for defensive copies or locks), but the feature itself isn't faster.|
|`for`-comprehension|Pure syntactic sugar. De-sugars into a series of `map`, `flatMap`, and `withFilter` calls before compilation.|Performance depends on the underlying collection, e.g., `List`.|
---

### Lazy Evaluation: Scala Collections vs. Java Streams

When performing operations on a standard Scala `List`, `Vector`, or `Seq`, each operation typically **creates a new, intermediate collection**.
This adds overheads.

```scala
val numbers = List(1, 2, 3, 4, 5, 6)

val result = numbers
  .filter(_ % 2 == 0) // Creates a NEW list: List(2, 4, 6)
  .map(_ * 10)       // Creates ANOTHER NEW list: List(20, 40, 60)
  .take(2)           // Creates a FINAL list: List(20, 40)
```

vs Java implementation that a Stream operation is **lazy**.
It doesn't do any work until a terminal operation (like `collect`, `forEach`, `findFirst`) is called.

```java
List<Integer> numbers = List.of(1, 2, 3, 4, 5, 6);

List<Integer> result = numbers.stream()
  .filter(n -> n % 2 == 0) // Doesn't create a new list. Just sets up a "filter" stage.
  .map(n -> n * 10)       // Doesn't create a new list. Just adds a "map" stage.
  .limit(2)               // Adds a "limit" stage.
  .collect(Collectors.toList()); // NOW the work happens.
```

Introduce Scala `.view` that prevents creating intermediate collections.

```scala
val numbers = List(1, 2, 3, 4, 5, 6)

val result = numbers
  .view         // <<<<<<< Creates a lazy view of the collection
  .filter(_ % 2 == 0)
  .map(_ * 10)
  .take(2)
  .toList       // <<<<<<< The terminal operation that forces evaluation
```

### Scala Performance Optimization on Immutable Objects with Design Pattern

Scala philosophy is "Things are assumed immutable unless declared mutable.", there are optimization considerations   on immutable objects.

For example, `user1` and `user2` are created at ease (just a few lines of code, builtin default `copy()`) and immutable (declared as `val`).

```scala
case class User(id: Long, name: String, email: String)

val user1 = User(1L, "Bob", "bob@example.com")
user1.name = "Robert" // <-- COMPILE ERROR!
                      // The 'name' field inside the case class is a val. It cannot be changed.

// How to "change" Bob's email? Use the free copy() method.
val user2 = user1.copy(email = "bob.new@example.com")

println(user1) // Prints: User(1,Bob,bob@example.com) -> Original is untouched
println(user2) // Prints: User(1,Bob,bob.new@example.com) -> New object with the change
```

The default assumption that data is immutable gives the implication that the data is READ only.
This makes design easy (lock-free) and performance can be boosted for data access is non-blocking.

## Scala Object-Oriented Programming (OOP) Development by `apply`

```scala
case class Animal(
  species: String, sound: String, details: Map[String, Any]
) {
  def description: Map[String, Any] = Animal.buildDescription(this)
}

object Animal {
  def buildDescription(animal: Animal): Map[String, Any] = {
    Map(
      "Species" -> animal.species,
      "Sound" -> animal.sound
    ) ++animal.details
  }

  def apply(cat: Cat) : Animal = {
    Animal (
      "Cat", "Meow", Map("Color" -> cat.color, "Length" -> cat.length)
    )
  }

  def apply(dog: Dog) : Animal = {
    Animal (
      "Dog", "Bark", Map("Breed" -> dog.breed, "Size" -> dog.size)
    )
  }
}

case class Cat(color:String, length: Int)
case class Dog(breed:String, size: Int)
```

### Companion Object

In scala, a *companion object* is a special object that shares the same name as its associated class and has access to its private members.

```scala
def apply(cat: Cat) : Animal = {
  Animal (
    "Cat", "Meow", Map("Color" -> cat.color, "Length" -> cat.length)
  )
}

def apply(dog: Dog) : Animal = {
  Animal (
    "Dog", "Bark", Map("Breed" -> dog.breed, "Size" -> dog.size)
  )
}
```

where `Animal` is the companion object that the inherited `Cat` and `Dog` can access the `Animal`'s private members.

By the design of companion object, scala can implement many programming paradigms, e.g., factory method, and many OOP designs.

## Scala Big Data Table Development by `Product`

## Scala Concurrency and The Akka Framework

Akka is a toolkit and runtime for building highly concurrent, distributed, and resilient applications on the Java Virtual Machine (JVM).
While it has Java and Scala APIs, it is most idiomatically used with Scala.

Akka provides a higher level of abstraction over threads and locks by *Actors*, avoided letting user manually put lock.

For example, a `Counter` is passed around different processes, and traditionally there will be frequent lock and unlock operation to sync the counter.
Here actors are used to manage the `Increment`.

```scala
object Counter {
  // Messages the actor can receive. These are immutable.
  case object Increment
  case object GetCount
}
```

Define how `Counter` is incremented.

```scala
import akka.actor.{Actor, ActorLogging}
import Counter._ // Import our messages

class Counter extends Actor with ActorLogging {

  // 1. STATE: This is the internal, private state of the actor.
  // It is safe because only this actor can modify it, one message at a time.
  var count = 0

  // 2. BEHAVIOR: This method defines how the actor reacts to messages.
  override def receive: Receive = {
    case Increment =>
      count += 1
      log.info(s"Count incremented to: $count")

    case GetCount =>
      // Reply to the original sender with the current count.
      // sender() is an ActorRef to the actor that sent the GetCount message.
      log.info(s"Received a request for the count. Replying with: $count")
      sender() ! count
  }
}
```

Finally, the object `Counter` (also singleton) is incremented without explicit lock implementation.
The output expects `3` since there are three invocations of `counterActor ! Increment`.

```scala
import akka.actor.{ActorSystem, Props}
import akka.pattern.ask
import akka.util.Timeout
import Counter._ // Import our messages

import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Failure, Success}


object MainApp extends App {

  // 1. Create the ActorSystem - the "home" for our actors
  val system = ActorSystem("CounterSystem")

  // 2. Create an instance of our Counter actor
  // Props is a configuration object for creating an actor.
  val counterActor = system.actorOf(Props[Counter], "my-counter")
  println("Created Counter Actor")

  // 3. Send "fire-and-forget" messages using the `!` (tell) operator
  println("Sending Increment messages...")
  counterActor ! Increment
  counterActor ! Increment
  counterActor ! Increment

  // Give the actor a moment to process the messages above
  Thread.sleep(1000)

  // 4. Send a message and expect a reply using `?` (ask)
  // `ask` returns a Future, as the reply will come back asynchronously.
  println("\nAsking for the final count...")
  implicit val timeout: Timeout = Timeout(3.seconds) // A timeout for the reply

  val futureResult = counterActor ? GetCount // Returns Future[Any]

  // Handle the result of the Future when it completes
  futureResult.onComplete {
    case Success(result) =>
      println(s"SUCCESS! The final count is: $result")
      // 5. Always shut down the actor system when you're done
      system.terminate()
    case Failure(ex) =>
      println(s"FAILURE! Could not get count. Reason: ${ex.getMessage}")
      system.terminate()
  }
}
```

### Akka Supervisor Strategy

A Supervisor Strategy is a parent actor's policy for handling failures in its child actors.
The below supervisor defines on what errors that if an actor should be stopped or resumed to next message process.

```scala
import akka.actor.SupervisorStrategy.{Escalate, Resume, Restart, Stop}
import akka.actor.{Actor, ActorLogging, OneForOneStrategy, Props, SupervisorStrategy}
import scala.concurrent.duration._

class Supervisor extends Actor with ActorLogging {

  // Create the child actor
  val dataParser = context.actorOf(Props[DataParser], "data-parser-child")

  // DEFINE THE SUPERVISOR STRATEGY
  override val supervisorStrategy: SupervisorStrategy =
    // OneForOneStrategy: The decision applies only to the failing child.
    OneForOneStrategy(maxNrOfRetries = 10, withinTimeRange = 1.minute) {
      // The "Decider": A PartialFunction that maps an Exception to a Directive.
      case _: IllegalStateException =>
        log.warning("Transient error detected. Resuming child.")
        Resume
      case _: RuntimeException =>
        log.error("Corrupted state detected. Restarting child.")
        Restart
      case _: IllegalArgumentException =>
        log.error("Fatal error detected. Stopping child.")
        Stop
      case _: Exception =>
        log.error("Unknown error. Escalating...")
        Escalate
    }

  override def receive: Receive = {
    // Forward any message it receives to its child
    case msg =>
      log.info(s"Supervisor received a message, forwarding to child: $msg")
      dataParser ! msg
  }
}
```

### Comparisons: Akka vs SpringBoot

* Spring Boot (MVC) primarily uses a Thread-Per-Request model.
* Akka uses the Actor Model with a shared pool of threads managed by a Dispatcher.

||Spring Boot (Web MVC)|Akka Actors (with Akka HTTP)|
|:---|:---|:---|
|Concurrency Model|Thread-Per-Request. A pool of threads where one thread handles one request from start to finish.|Actor Model. A small, shared pool of threads serves many lightweight actors via message passing.|
|Execution Style|Synchronous / Blocking by default. A thread waits during I/O operations.|Asynchronous / Non-Blocking by nature. A thread is released during I/O and used for other work.|
|Use Scenarios|Very easy for standard CRUD/REST APIs due to strong conventions and annotations.|Message passing.|