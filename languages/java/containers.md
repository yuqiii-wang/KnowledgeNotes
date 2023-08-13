# Java Primitives and Containers

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

Immutable char array is good for constant hash and multi-threading, 
and is stored in constant memory area in JVM as cache so that next time the char array's constructed/destroyed/read, there is no need of reallocating memory.

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

Since Java 9, `char[]` is changed to `byte[]` to save memory for char storage.

## Java Collections

Java collection provides common data structures and algorithms that store objects.

<div style="display: flex; justify-content: center;">
      <img src="imgs/java_collections.png" width="60%" height="60%" alt="java_collections" />
</div>
</br>

### ArrayList

The ArrayList (`std::vector` in c++) class maintains the insertion order and is non-synchronized. 
The elements stored in the ArrayList class can be randomly accessed. 

```java
import java.util.*;  
class TestJavaCollection1{  
  public static void main(String args[]){  

    ArrayList<String> list=new ArrayList<String>();//Creating an arraylist 
 
    list.add("Jack");//Adding object in the arraylist  
    list.add("Jason");  
    list.add("Jay");  
    list.add("Jason");  

    //Traversing list through Iterator  
    Iterator itr=list.iterator();  
    while(itr.hasNext()){  
      System.out.println(itr.next());  
    }  
  }  
}  
```

### LinkedList

It uses a doubly linked list internally to store the elements. 
It maintains the insertion order and is not synchronized. 

### Vector

Vector uses a dynamic array to store the data elements. It is similar to ArrayList. However, It is synchronized and contains many methods that are not the part of Collection framework.

### Stack and Queue

The stack is the subclass of Vector. It implements the **last-in-first-out** data structure.

Queue interface maintains the **first-in-first-out** order. 

```java
import java.util.*;  
public class TestJavaCollection{  
  public static void main(String args[]){  

    Stack<String> stack = new Stack<String>();  

    stack.push("Jason");  
    stack.push("Jay");  
    stack.push("Jayden");  
    stack.push("Jack");  
    stack.push("Jackson");  
    stack.pop();  // after pop, "Jackson" is removed

    Iterator<String> itr=stack.iterator();  
    while(itr.hasNext()){  
      System.out.println(itr.next());  
    }  
  }  
}  
```

### HashSet

By hash, in the below code only two items are stored in `set`.

```java
import java.util.*;  
class TestJavaCollection1{  
  public static void main(String args[]){  

    HashSet<String> set = new HashSet<String>();//Creating an HashSet 
 
    set.add("Jack"); // Adding object in the HashSet  
    set.add("Jason");  
    set.add("Jason");  // this item is duplicate and lost

    //Traversing set through Iterator  
    Iterator itr = set.iterator();  
    while(itr.hasNext()){  
      System.out.println(itr.next());  
    }  
  }  
}  
```

### HashMap vs HashTable

HashMap is non-synchronized and not thread-safe, can’t be shared between many threads without proper mutex,
whereas Hashtable is synchronized, thread-safe and can be shared with many threads.

HashMap allows one null key and multiple null values whereas Hashtable doesn’t allow any null key or value.

HashMap is generally preferred over HashTable if thread synchronization is not needed.

```java
// Java program to demonstrate
// HashMap and HashTable
import java.util.*;
import java.lang.*;
import java.io.*;
 
// Name of the class has to be "Main"
// only if the class is public
class JavaHashMapTable
{
    public static void main(String args[])
    {
        //----------hashtable -------------------------
        Hashtable<Integer,String> ht=new Hashtable<Integer,String>();
        ht.put(101,"Jason");
        ht.put(102,"Amy");
        ht.put(102,"Jack"); // value updated to Jack
        System.out.println("-------------Hash table--------------");
        for (Map.Entry m:ht.entrySet()) {
            System.out.println(m.getKey()+" "+m.getValue());
        }
 
        //----------------hashmap--------------------------------
        HashMap<Integer,String> hm=new HashMap<Integer,String>();
        hm.put(100,"Jason");
        hm.put(104,"Amy");
        hm.put(104,"Jack"); // value updated to Jack
        System.out.println("-----------Hash map-----------");
        for (Map.Entry m:hm.entrySet()) {
            System.out.println(m.getKey()+" "+m.getValue());
        }
    }
}
```

* Why HashMap is not thread-safe:

A hash map is based on an array, where each item represents a bucket. 
As more keys are added, the buckets grow and at a certain threshold the array is recreated with a bigger size, its buckets rearranged so that they are spread more evenly (performance considerations). 
It means that sometimes `HashMap#put()` will internally call `HashMap#resize()` to make the underlying array bigger. `HashMap#resize()` assigns the table field a new empty array with a bigger capacity and populates it with the old items. During re-polulation, when a thread accesses this HashMap, this HashMap may return `null`.

```java
final Map<Integer, String> map = new HashMap<>();

final Integer targetKey = 0b1111_1111_1111_1111; // 65 535, forced JVM to resize and populate
final String targetValue = "v";
map.put(targetKey, targetValue);

new Thread(() -> {
    IntStream.range(0, targetKey).forEach(key -> map.put(key, "someValue"));
}).start(); // start another thread to add key/value pairs


while (true) {
    if (!targetValue.equals(map.get(targetKey))) {
        throw new RuntimeException("HashMap is not thread safe."); // throw err
    }
}
```
