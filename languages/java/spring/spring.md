# Spring

## Quick Intro: Spring vs Spring Boot vs Spring Cloud

### Spring Framework

* Build on core java for web application
* Provide Inversion of Control (IoC) and Dependency Injection (DI)
* Aspect-Oriented Programming (AOP): a programming paradigm that aims to increase modularity by allowing the separation of cross-cutting concerns. In Spring Framework, it addresses concerns such as logging, transaction management, security, etc., separately from the business logic.
* provides a comprehensive transaction management JDBC, JPA, JTA, etc.
* Spring MVC (Model-View-Controller): built on top of the core Spring framework

### Spring Boot

* Simplifies Spring application development with auto-configuration
* Includes embedded servers (like Tomcat or Jetty)

### Spring Cloud

* Build on Spring Boot
* For distributed systems and microservices, especially in cloud environments
* Characterized by service discovery, circuit breakers, intelligent routing, distributed sessions, etc.

## Spring Context and Inversion of Control (IoC)

When java objects are under spring context, they are called *beans* and their lifecycles and dependencies are managed by spring.

Inversion of Control (IoC) refers to the process where the control of object creation and the management of their dependencies is transferred from the application code to a container or framework.

Spring uses `@Bean` to take control of an object, such as setting its member values and managing object life cycle.

A spring context starts with `@SpringBootApplication`, or in test `@SpringBootTest`.

```java
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class MySpringBootApplication {
    public static void main(String[] args) {
        SpringApplication.run(MySpringBootApplication.class, args);
    }
}
```

Beans can be defined using annotations like `@Component`, `@Service`, `@Repository`, `@Controller`, or `@Bean` methods in @Configuration classes.

### Reflection

In Java, reflection can be used to directly "retrieve"/"construct" a class or the class' member, bypassing `private` scope and just creating an object without having even known this class.

For example, below code shows directly accessing `java.util.ArrayList`'s member function and creating a new class without explicitly invoking the constructor.

```java
import java.lang.reflect.Constructor;

public class ReflectionExample {
    public static void main(String[] args) throws Exception {
        Class<?> clazz = Class.forName("java.util.ArrayList");
        System.out.println("Class name: " + clazz.getName());

        // Create a new instance of ArrayList
        Constructor<?> constructor = clazz.getConstructor();
        Object obj = constructor.newInstance();
        System.out.println("Created object: " + obj);
    }
}
```

In Spring Boot, reflection is used in the context of *dependency injection* (DI), where the framework needs to create and wire together beans.

### Dependency Injection (DI)

Dependency Injection (or sometime called wiring) helps in gluing independent classes together and at the same time keeping them independent (decoupling).

```java
// shouldn't be this
public class TextEditor {
   private SpellChecker spellChecker;
   public TextEditor() {
      spellChecker = new SpellChecker();
   }
}

// instead, should be this, so that regardless of changes of SpellChecker class, 
// there is no need of changes to SpellChecker object implementation code 
public class TextEditor {
   private SpellChecker spellChecker;
   public TextEditor(SpellChecker spellChecker) {
      this.spellChecker = spellChecker;
   }
}
```

where the input arg `SpellChecker spellChecker` should be passed in (likely already instantiated elsewhere) rather than be `new()` every time.

### IoC container

In the Spring framework, the interface ApplicationContext represents the IoC container. The Spring container is responsible for instantiating, configuring and assembling objects known as *beans*, as well as managing their life cycles.

In order to assemble beans, the container uses configuration metadata, which can be in the form of XML configuration or annotations, such as setting up attributes for this bean in `applicationContext.xml`, which is loaded by `ClassPathXmlApplicationContext`.

```java
ApplicationContext context
  = new ClassPathXmlApplicationContext("applicationContext.xml");
```

### Dependency Injection Examples

* bean-Based

If we don't specify a custom name, then the bean name will default to the method name.

```java
@Configuration
public class TextEditor {

   private SpellChecker spellChecker;

   @Bean
   public TextEditor() {
      spellChecker = new SpellChecker();
   }
}
```

#### `@Autowired`

`@Autowired` is used to perform injection

When the Spring context is initialized, it starts creating instances of the beans defined in the configuration (either XML or Java-based configuration).

It scans for classes annotated with `@Component`, `@Service`, `@Repository`, and other stereotype annotations, which are automatically registered as Spring beans.

For each bean, Spring checks for the `@Autowired` annotation on constructors, and tries to resolve and inject all the constructor parameters (also same to field and setter).

Spring uses reflection to set the field values or call the constructor/setter methods with the resolved dependencies.
The ApplicationContext (or BeanFactory) is responsible for managing the lifecycle and scope of beans and resolving dependencies as needed.

* P.S. `@Autowired` is only triggered when in `@SpringApplicationContext` (as well as in `@SpringBootTest`).

## Config

### `@Value` and `application.properties`

Spring applications by default load from `application.properties`, where items are auto mapped in spring framework via `@Value`.

For example, in `application.properties`

```conf
greeting.message=Hello World!
```

The `greeting.message` is retrievable in

```java
package example.springvalue.annotation.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class ValueController {

    @Value("${greeting.message}") 
    private String greetingMessage;

    @GetMapping("")
    public String sendGreeting(){
        return greetingMessage;
    }
}
```

To prevent error

```txt
java.lang.IllegalArgumentException: Could not resolve placeholder 'greeting.message' in value "${greeting.message}"
```

one can use `@Value("${greeting.message:Greeting not found!}")`, where `:` is an alternative value in case of `greeting.message` not found.

### `@Profile`

`@Profile` allows user to map beans to different profiles, typically diff envs, e.g., dev, test, and prod.

In `application.properties`, config the env.

```conf
spring.profiles.active=dev
```

In implementation, only used in `dev`.

```java
@Component
@Profile("dev")
public class DevDatasourceConfig { ... }
```

or `dev` is NOT active.

```java
@Component
@Profile("dev")
public class NonDevDatasourceConfig { ... }
```

### `@Configuration`, and `@bean` vs `@Component`

#### `@Configuration` and `@bean`

`@Bean` is used within a `@Configuration` class to explicitly declare a bean.
`@bean` is primitive compared to `@Component`, hence provided fine-grained control over instantiation.

In Spring, instantiated beans have a `singleton` scope by default.
This is problematic, as exampled in below when `clientDao()` is called once in `clientService1()` and once in `clientService2()`, but only one singleton instance is returned.

`@Configuration` comes in rescue that beans under `@Configuration`-annotated `AppConfig` will see instantiations of two beans.

```java
@Configuration
public class AppConfig {

    @Bean
    public ClientService clientService1() {
        ClientServiceImpl clientService = new ClientServiceImpl();
        clientService.setClientDao(clientDao());
        return clientService;
    }

    @Bean
    public ClientService clientService2() {
        ClientServiceImpl clientService = new ClientServiceImpl();
        clientService.setClientDao(clientDao());
        return clientService;
    }

    @Bean
    public ClientDao clientDao() {
        return new ClientDaoImpl();
    }
}
```

#### `@bean` vs `@Component`

* `@Component`

Be automatically detected and managed by Spring.

For application-specific classes such as services, repositories, and controllers.

* `@bean`

Need to configure beans for third-party libraries or have fine-grained control over bean instantiation.

## Spring IOC Loading Process
