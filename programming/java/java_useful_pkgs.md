# Java Useful Packages

## Useful Utils

* `BeansUtils` for Copy
* `Jackson` for Json

## Concurrency and `CompletableFuture`

Concurrency is useful that for only given one thread, such thread can do other tasks (Operating System (OS) can assign other tasks to this thread) while waiting for an operation to finish.
This is different from multi-threading that such a thread is explicitly set idle waiting (but from OS perspective, this thread is still busy not assignable) for the result of an operation.

Such concept is termed *future*.

In practice, *promise* is used for it contains more utility functions.

|Feature|Future (Java)|Promise (e.g., JavaScript, `CompletableFuture`)|
|-|-|-|
|Completion|Set by the task itself|Set manually (`resolve`/`reject`)|
|Chaining|No built-in chaining|Chaining supported with `.then()`, `.catch()` or `thenApply()`, `exceptionally()`|
|Result Handling|No manual result modification|Manual resolution/rejection (`resolve()`, `reject()`)|

In Java implementation, `CompletableFuture` is a popular tool of the Promise.

```java
import java.util.concurrent.*;

public class CompletableFutureExample {
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        CompletableFuture<Integer> completableFuture = CompletableFuture.supplyAsync(() -> {
            try {
                Thread.sleep(2000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return 42;
        });
        
        // Do other work here
        completableFuture.thenAccept(result -> {
            System.out.println("Result: " + result); // Handling success
        }).exceptionally(e -> {
            System.out.println("Error: " + e.getMessage());
            return null;
        });
        
        // Block to ensure the program doesn't exit before the task is complete
        completableFuture.get();
    }
}
```

## Auto Java Method Generation

### `lombok.Data`

`@Data` is a convenient annotation used to automatically generate boilerplate code.
Annotate a class with `@Data`, Lombok generates the following:

* Getters: For all non-static fields.
* Setters: For all non-final, non-static fields.
* `toString()`: A method that returns a string representation of the object.
* `equals()` and `hashCode()`: Methods to compare objects based on their fields.
* Constructor: A constructor that initializes all final or `@NonNull` fields.

For example,

```java
import lombok.Data;

@Data
public class Person {
    private String name;
    private int age;
    private final String id; // final field, included in constructor
}
```

`lombok.Data` will populate methods such as

```java
public class Person {
    ...

    @Override
    public String toString() {
        return "Person(name=" + this.name + ", age=" + this.age + ", id=" + this.id + ")";
    }

    @Override
    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        result = result * PRIME + this.getAge();
        result = result * PRIME + (this.getName() == null ? 43 : this.getName().hashCode());
        result = result * PRIME + (this.getId() == null ? 43 : this.getId().hashCode());
        return result;
    }
}
```

## Lightweight Micro Framework

A micro-framework is built on top of servlet to handle http requests/responses, and it is lightweight.
Spark Java is an HTTP REST lightweight micro-framework.

```java
// Import the static methods from Spark
import static spark.Spark.*;

public class SparkExample {
    public static void main(String[] args) {

        // Define a route for GET requests to "/hello"
        get("/hello", (request, response) -> {
            // 'request' is an instance of spark.Request
            // Retrieve a query parameter named "name" if provided (e.g., /hello?name=Alice)
            String name = request.queryParams("name");

            // 'response' is an instance of spark.Response
            // Set the content type of the response
            response.type("text/html");

            // Return a personalized greeting if a name was provided; otherwise, a default message
            if (name != null && !name.isEmpty()) {
                return "<h1>Hello, " + name + "!</h1>";
            } else {
                return "<h1>Hello, World!</h1>";
            }
        });

        // Another example: a POST endpoint that echoes back the request body
        post("/echo", (request, response) -> {
            // Get the raw body of the request
            String body = request.body();
            
            // Set the response type to plain text
            response.type("text/plain");
            
            // Optionally set a custom header
            response.header("X-Custom-Header", "SparkExample");
            
            // Return the same body as the response
            return "Received: " + body;
        });
    }
}
```

## Logging by Log4j

Log4j is a popular java logging tool.
Example is such as below

```java
package com.example.dao;

import org.apache.log4j.Logger;

public class UserDao {

    // Get a logger instance for this specific class.
    private static final Logger logger = Logger.getLogger(UserDao.class);

    public void findUser(int userId) {
        logger.info("Attempting to find user with ID: " + userId);
    }
}
```

A log4j.properties file is structured around three main components:

* Loggers: corresponds to java class to pinpoint logging control.
* Appenders: responsible for determining the destination of the log messages, console, files, databases, and more.
* Layouts: specify what information is included in the log output, such as the timestamp, logging level, class name, and the actual log message.

For example,

```txt
src
├── com
│   └── example
│       ├── MainApp.java
│       ├── dao
│       │   └── UserDao.java
│       └── service
│           └── OrderService.java
└── log4j.properties
```

Then in below config `.properties` file,
`log4j.logger.com.example.dao=DEBUG, DAO_FILE` and `log4j.logger.com.example.service=DEBUG, SERVICE_FILE` are sued to bind java class vs config

```properties
# Root logger prints to the console at INFO level.
# This will catch logs from any class not specifically configured otherwise.
log4j.rootLogger=INFO, CONSOLE

# --- Console Appender Configuration ---
# All logs at INFO level or higher from any logger will also go here,
# unless additivity is turned off for that logger.
log4j.appender.CONSOLE=org.apache.log4j.ConsoleAppender
log4j.appender.CONSOLE.layout=org.apache.log4j.PatternLayout
log4j.appender.CONSOLE.layout.ConversionPattern=%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n

# --- DAO File Appender Configuration (for UserDao) ---
# This appender will write to dao.log
log4j.appender.DAO_FILE=org.apache.log4j.RollingFileAppender
log4j.appender.DAO_FILE.File=logs/dao.log
log4j.appender.DAO_FILE.MaxFileSize=5MB
log4j.appender.DAO_FILE.MaxBackupIndex=5
log4j.appender.DAO_FILE.layout=org.apache.log4j.PatternLayout
log4j.appender.DAO_FILE.layout.ConversionPattern=%d{yyyy-MM-dd HH:mm:ss} [%t] %-5p %c - %m%n

# --- Service File Appender Configuration (for OrderService) ---
# This appender will write to service.log
log4j.appender.SERVICE_FILE=org.apache.log4j.RollingFileAppender
log4j.appender.SERVICE_FILE.File=logs/service.log
log4j.appender.SERVICE_FILE.MaxFileSize=5MB
log4j.appender.SERVICE_FILE.MaxBackupIndex=5
log4j.appender.SERVICE_FILE.layout=org.apache.log4j.PatternLayout
log4j.appender.SERVICE_FILE.layout.ConversionPattern=%d{yyyy-MM-dd HH:mm:ss} [%t] %-5p %c - %m%n


# --- Logger Configuration for DAO Package ---
# Logger for any class in the 'com.example.dao' package.
# It will log at DEBUG level and send output to the DAO_FILE appender.
log4j.logger.com.example.dao=DEBUG, DAO_FILE
# This is crucial: it prevents logs from this logger from also being sent
# to the rootLogger's appenders (i.e., the CONSOLE).
log4j.additivity.com.example.dao=false


# --- Logger Configuration for Service Package ---
# Logger for any class in the 'com.example.service' package.
# It will log at DEBUG level and send output to the SERVICE_FILE appender.
log4j.logger.com.example.service=DEBUG, SERVICE_FILE
# Prevent these logs from also going to the CONSOLE.
log4j.additivity.com.example.service=false
```