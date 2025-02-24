# Scala

Scala specializes data model templating.

## Basic Syntax to Scala

### Immutable and Mutable Variables: `val` vs. `var`

* `val`: Defines an immutable variable. Once assigned, its value cannot change.
* `var`: Defines a mutable variable. Its value can be reassigned.

### Classes vs. Objects

* Classes in Scala are blueprints for creating objects (instances).

```scala
class Person(val name: String, var age: Int) {
  def greet(): Unit = println(s"Hello, my name is $name and I am $age years old.")
}

// Instantiation:
val alice = new Person("Alice", 30)
alice.greet()  // Output: Hello, my name is Alice and I am 30 years old.
```

* Objects are **singletons**. They are created lazily on first access and ensure there is only one instance.

```scala
object Person {
  // Factory method to create a Person without explicitly using "new"
  def apply(name: String, age: Int): Person = new Person(name, age)
}

// Usage:
val bob = Person("Bob", 25)  // No need to call "new"
bob.greet()  // Output: Hello, my name is Bob and I am 25 years old.
```

#### Case Classes and Pattern Matching

Case classes are used for pattern matching then trigger execution of corresponding parts of code.

* Auto-generated methods like `equals`, `hashCode`, `toString`, and a `copy` method.
* They are immutable by default

For example, particular conditions `Point(0, 0)` and `Point(x, y)` are associated with lambda expression, respectively.

```scala
case class Point(x: Int, y: Int)

// Creating an instance (no need for "new")
val p = Point(2, 3)

p match {
  case Point(0, 0) => println("The point is at the origin.")
  case Point(x, y) => println(s"The point is at ($x, $y).")
}
```

#### `sealed` for Controlled Inheritance

When marked a `class` or trait as `sealed`, all of its direct subclasses must be declared in the same file.
This is useful for inheritance and case pattern match management.

For example, having declared `sealed trait Expr`, all inherited `case class`es must be declared.

```scala
sealed trait Expr

case class Num(value: Int) extends Expr
case class Add(lhs: Expr, rhs: Expr) extends Expr
case class Mul(lhs: Expr, rhs: Expr) extends Expr

def eval(expr: Expr): Int = expr match {
  case Num(value) => value
  case Add(lhs, rhs) => eval(lhs) + eval(rhs)
  case Mul(lhs, rhs) => eval(lhs) * eval(rhs)
  // If you were to omit one of the cases, the compiler might warn you that the match is not exhaustive.
}
```

### Traits

Traits in Scala are similar to Java interfaces but can also provide concrete method implementations and maintain state.

```scala
trait Greeter {
  def greet(): Unit = println("Hello!")
}

class FriendlyPerson(val name: String) extends Greeter {
  // Optionally, override the default implementation
  override def greet(): Unit = println(s"Hi, I'm $name. Nice to meet you!")
}

val charlie = new FriendlyPerson("Charlie")
charlie.greet()  // Output: Hi, I'm Charlie. Nice to meet you!
```

### Template Typing Bound and Casting

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
