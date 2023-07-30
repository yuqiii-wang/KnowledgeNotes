# Java Practices

## `concurrentHashMap` thread safety

## Comparison between `int` and `integer`

## `hashcode` overload with `equals`

## `aop` concept

## `jvm` annotations

## Basic Java Data Types and Wrapper

In java, every thing is an object, even for the basic data types.

|Primitive|Wrapper|
|-|-|
|boolean | Boolean |
|byte | Byte |
|char | Character |
|float | Float |
|int | Integer |
|long | Long |
|short | Short |
|double | Double |

For example:
```java
Integer x = 1; // wrapping Integer.valueOf(1)
int y = x; // invoked X.intValue()
```

String is immutable/final for the stored data in `final char value[];`.
When a new string is constructed (via concatenation or some other methods), the original string object actually constructs a new `final char value[];` then points to the new char array.

Immutable char array is good for constant hash and multi-threading.

```java
public final class String
    implements java.io.Serializable, Comparable<String>, CharSequence {
    /** The value is used for character storage. */
    private final char value[];
​
    /** Cache the hash code for the string */
    private int hash; // Default to 0
}
```

The number of constructed objects by `new String("hello")` is two:
* `"hello"` is constructed at compile time stored as a char string object in a constant var pool
* `new` constructs  string object in heap

## Number 