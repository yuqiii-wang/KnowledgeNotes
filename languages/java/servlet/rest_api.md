# REST API

The Representational State Transfer (REST) is an HTTP-based architectural style where APIs sit there awaiting HTTP requests, usually in `GET`, `POST`, etc. with rich info passed in via a json-like format.
This is different from typical software APIs where args are fixed.

```java
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;

public class API {
    public static void main(String[] args) throws IOException {

        HttpServer server = HttpServer.create(new InetSocketAddress(8080), 0);
        server.createContext("/api/greeting", (exchange -> {

            if ("GET".equals(exchange.getRequestMethod())) {
                String responseText = "Hello World! from our framework-less REST API\n";
                exchange.sendResponseHeaders(200, responseText.getBytes().length);
                OutputStream output = exchange.getResponseBody();
                output.write(responseText.getBytes());
                output.flush();
            } else {
                exchange.sendResponseHeaders(405, -1);// 405 Method Not Allowed
            }
            exchange.close();
        }));


        server.setExecutor(null); // creates a default executor
        server.start();

    }
}
```

## Jersey (JAX-RS)

Jersey is an open-source framework that provides a reference implementation of JAX-RS (Java API for RESTful Web Services).

For example, build the application that by `GET` http://localhost:8080/api/hello, The response will be "Hello, Jersey!".

1. Create the Resource Class (define the REST API)

```java
import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;

@Path("/hello")
public class HelloResource {

    @GET
    @Produces("text/plain")
    public String sayHello() {
        return "Hello, Jersey!";
    }
}
```

2. Configure the Application Class:

```java
import javax.ws.rs.ApplicationPath;
import javax.ws.rs.core.Application;

@ApplicationPath("/api") // Base URI for the REST API
public class MyApplication extends Application {
    // No additional code needed, Jersey scans the package for resources
}
```

### Handling JSON in Jersey with Jackson

By default, Jersey supports JSON binding through the Jackson library.

1. Define the data schema

```java
import com.fasterxml.jackson.annotation.JsonProperty;

public class Person {
    @JsonProperty("name")
    private String name;

    @JsonProperty("age")
    private int age;

    // Getters and Setters
}
```

2. Define the REST API that serializes the object into a json

```java
import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;

@Path("/person")
public class PersonResource {

    @GET
    @Produces("application/json")
    public Person getPerson() {
        Person person = new Person();
        person.setName("John Doe");
        person.setAge(30);
        return person;
    }
}
```

### JAX-RS Client HTTP Request by `Entity`

`javax.ws.rs.client.Entity` is a class in the JAX-RS Client API that represents a request entity or a response entity when communicating with a RESTful web service.

Below example uses `Invocation.Builder` to configure HTTP requests with details like HTTP headers, authentication, or media types before making the request.
Finally, the HTTP request is sent by `invocationBuilder.post(entity);`.

```java
import javax.ws.rs.client.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import javax.ws.rs.core.HttpHeaders;

public class JAXRSClientWithInvocationBuilder {

    public static void main(String[] args) {
        // Create a JAX-RS client
        Client client = ClientBuilder.newClient();

        // Create the target URI (RESTful API endpoint)
        WebTarget target = client.target("http://example.com/api/people");

        // Create the Person object to send
        Person person = new Person("John Doe", 30);

        // Create the Entity (the request payload) and set the media type to JSON
        Entity<Person> entity = Entity.entity(person, MediaType.APPLICATION_JSON);

        // Create an Invocation.Builder to customize the request
        Invocation.Builder invocationBuilder = target.request(MediaType.APPLICATION_JSON);

        // Set custom headers using the Invocation.Builder (e.g., adding an authorization header)
        invocationBuilder.header(HttpHeaders.AUTHORIZATION, "Bearer your_token_here");
        invocationBuilder.header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON);

        // Send the POST request with the entity (payload) and headers
        Response response = invocationBuilder.post(entity);

        // Check if the request was successful (HTTP 201 Created)
        if (response.getStatus() == 201) {
            System.out.println("Person created successfully!");

            // Optionally, read the response entity (e.g., the created person data)
            Person createdPerson = response.readEntity(Person.class);
            System.out.println("Created Person: " + createdPerson.getName() + ", Age: " + createdPerson.getAge());
        } else {
            System.out.println("Failed to create Person. Status: " + response.getStatus());
        }

        // Close the response and client
        response.close();
        client.close();
    }
}
```

### Jersey vs Spring

Jersey and Spring Boot both provide frameworks for building RESTful web services in Java.

* Ecosystem and capability

Jersey is a lightweight framework focused primarily on creating RESTful web services.

Spring Boot (Spring MVC) is much heavier provided services such as servlet dispatch and more.

* Development Approach

Jersey offers a more traditional Java EE approach. It's typically used in enterprise applications where JAX-RS is preferred for building REST APIs.

Spring Boot encourages a microservice-based, modular approach.

* Maintainer

Jersey is maintained by Eclipse (previously by *GlassFish*)

Spring Boot has a huge community and a vast ecosystem.
