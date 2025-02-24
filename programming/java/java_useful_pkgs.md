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