# Java Practices

## `concurrentHashMap` thread safety

## `aop` concept

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

## Typical Object Methods

The `Object` class is the parent class of all the classes in java by default. 
It provides useful methods such as `toString()`. 
It is defined in `Java.lang.Object`.

### `equals()` vs `==`, and `hashCode`

By default, the hash of an object is computed by its memory location.

```java
public class Cat {
    public static void main(String[] args) {
        System.out.println(new Cat().hashCode());
    }
    //out
    //1349277854
}
```

Use `==` operators for reference comparison (address comparison) and `.equals()` method for content comparison. 
In simple words, `==` checks if both objects point to the same memory location whereas `.equals()` evaluates to the comparison of values in the objects.

```java
public class Test {
    public static void main(String[] args)
    {
        String s1 = "HELLO";
        String s2 = "HELLO";
        String s3 =  new String("HELLO");
 
        System.out.println(s1 == s2); // true
        System.out.println(s1 == s3); // false
        System.out.println(s1.equals(s2)); // true
        System.out.println(s1.equals(s3)); // true
    }
}
```

Both `==` and `equals()` can be overriden.

* `equals()` vs `hashCode()`

Objects that are equal (according to their `equals()`) must return the same hash code.

`hashCode()` has usage in hash map serving as the key.

### `getClass()`, `forName(...)` and Reflection

`getClass()` gets object's class.

```java
public class Person {
    private String name;

    public Person(String name) {
        this.name = name;
    }

    public static void main(String[] args) {
        Person p = new Person("Jack");
        Class clz = p.getClass();
        System.out.println(clz);
        System.out.println(clz.getName());
    }
    /**
     * class com.tyson.basic.Person
     * com.tyson.basic.Person
     */
}
```

`forName()` method of `java.lang.Class` class is used to get the instance of this Class with the specified class name. 
This class name is specified as the string parameter.

```java
public class Test {
    public static void main(String[] args)
        throws ClassNotFoundException
    {
        // get the Class instance using forName method
        Class c1 = Class.forName("java.lang.String");
 
        System.out.print("Class represented by c1: "
                         + c1.toString());
    }
}
```

Together, `getClass()` and `forName(...)` can be used in reflection to dynamically load classes/objects.

For example, in spring, objects are launched by an `xml` file that details what properties the object should have.

```java
public class BeanFactory {
       private Map<String, Object> beanMap = new HashMap<String, Object>();

       public void init(String xml) {
              try {
                     // read xml file and parse the xml, extract elements
                     SAXReader reader = new SAXReader();
                     ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
                     InputStream ins = classLoader.getResourceAsStream(xml);
                     Document doc = reader.read(ins);
                     Element root = doc.getRootElement();  
                     Element foo;

                     // iterate specified beans in the xml file
                     for (Iterator i = root.elementIterator("bean"); i.hasNext();) {  
                            foo = (Element) i.next();

                            // read from beans then get the class
                            Attribute id = foo.attribute("id");  
                            Attribute cls = foo.attribute("class");
                            Class bean = Class.forName(cls.getText());

                            // further get the class's info
                            java.beans.BeanInfo info = java.beans.Introspector.getBeanInfo(bean);
                            java.beans.PropertyDescriptor pd[] = info.getPropertyDescriptors();
                            Method mSet = null;
                            
                            // launch a new inst for the class
                            Object obj = bean.newInstance();

                            // read properties of the inst and set the values
                            for (Iterator ite = foo.elementIterator("property"); ite.hasNext();) {  
                                   Element foo2 = (Element) ite.next();
                                   // get property name
                                   Attribute name = foo2.attribute("name");
                                   String value = null;

                                   // read property value
                                   for(Iterator ite1 = foo2.elementIterator("value"); ite1.hasNext();) {
                                          Element node = (Element) ite1.next();
                                          value = node.getText();
                                          break;
                                   }

                                   for (int k = 0; k < pd.length; k++) {
                                          if (pd[k].getName().equalsIgnoreCase(name.getText())) {
                                                 mSet = pd[k].getWriteMethod();
                                                 // set the property value to the inst
                                                 mSet.invoke(obj, value);
                                          }
                                   }
                            }

                            // finished and put the object into the bean map
                            beanMap.put(id.getText(), obj);
                     }
              } catch (Exception e) {
                     System.out.println(e.toString());
              }
       }

       //other codes
}
```

## Numbers and Default Values

## `final`, `finally` and `finalize`

`final` is used to declare a variable as immutable. 
Similar to c++'s `const`, except that `final` cannot be inherited.

`finally` is used in `try`-`catch`-`finally` logic.

`finalize` is a method of `Object`, typically used in garbage collection.

## Common Exceptions and Examples

* `ClassCastException`

```java
Animal a = new Dog();
Dog d = (Dog) a; // No problem, the type animal can be casted to a dog, because it's a dog.
Cat c = (Dog) a; // Will cause a compiler error for type mismatch; you can't cast a dog to a cat.
```

* `NullPointerException`

It happens when dealing with null object.

For example, in the code below, `ptr.equals("gfg")` throws exception for `ptr` is null.
By simply changing it to `"gfg".equals(ptr)`, the err is solved.

```java
import java.io.*;
 
class GFGWrong
{
    public static void main (String[] args)
    {
        // Initializing String variable with null value
        String ptr = null;
 
        // Checking if ptr.equals null or works fine.
        try
        {
            // This line of code throws NullPointerException
            // because ptr is null
            if (ptr.equals("gfg"))
                System.out.print("Same");
            else
                System.out.print("Not Same");
        }
        catch(NullPointerException e)
        {
            System.out.print("NullPointerException Caught");
        }
    }
}
 
class GFGCorrect
{
    public static void main (String[] args)
    {
        // Initializing String variable with null value
        String ptr = null;
 
        // Checking if ptr is null using try catch.
        try
        {
            if ("gfg".equals(ptr))
                System.out.print("Same");
            else
                System.out.print("Not Same");           
        }
        catch(NullPointerException e)
        {
            System.out.print("Caught NullPointerException");
        }
    }
}
```

* `ArrayStoreException`

Thrown to indicate that an attempt has been made to store the wrong type of object into an array of objects. 
For example, the following code generates an ArrayStoreException:

```java
Object x[] = new String[3];
x[0] = new Integer(0);
```

## Blocking and Non-Blocking vs Sync and Async

## Java Threading and Runnable

`java.lang.Runnable` is an interface that is to be implemented by a class whose instances are intended to be executed by a thread. 

```java
public class RunnableDemo {
 
    public static void main(String[] args) {
        System.out.println("Main thread is- "
                        + Thread.currentThread().getName());
        Thread t1 = new Thread(new RunnableDemo().new RunnableImpl());
        t1.start();
    }
 
    private class RunnableImpl implements Runnable {
 
        public void run() {
            System.out.println(Thread.currentThread().getName()
                             + ", executing run() method!");
        }
    }
}
```
that outputs 
```txt
Main thread is- main
Thread-0, executing run() method!
```