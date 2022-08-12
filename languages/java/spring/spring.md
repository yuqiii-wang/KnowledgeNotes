# Spring

## Inversion of Control (IoC)

IoC transfers the control of objects or portions of a program to a container or framework; it enables a framework to take control of the flow of a program and make calls to our custom code.

In other words, spring uses `@Bean` to take control of an object, such as setting its member values and managing object life cycle.

### Dependency Injection

Dependency Injection (or sometime called wiring) helps in gluing independent classes together and at the same time keeping them independent (decoupling).

```java
// shouldn't be this
public class TextEditor {
   private SpellChecker spellChecker;
   public TextEditor() {
      spellChecker = new SpellChecker();
   }
}

// instead, should be this, so that regardless of changes of SpellChecker class, there is no need of changes to SpellChecker object implementation code 
public class TextEditor {
   private SpellChecker spellChecker;
   public TextEditor(SpellChecker spellChecker) {
      this.spellChecker = spellChecker;
   }
}
```

### IoC container

In the Spring framework, the interface ApplicationContext represents the IoC container. The Spring container is responsible for instantiating, configuring and assembling objects known as *beans*, as well as managing their life cycles.

In order to assemble beans, the container uses configuration metadata, which can be in the form of XML configuration or annotations, such as setting up attributes for this bean in `applicationContext.xml`, which is loaded by `ClassPathXmlApplicationContext`
```java
ApplicationContext context
  = new ClassPathXmlApplicationContext("applicationContext.xml");
```

### Dependency Injection materialization

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

* Autowire

Wiring allows the Spring container to automatically resolve dependencies between collaborating beans by inspecting the beans that have been defined.

```java
public class TextEditor {

   @Autowired
   private SpellChecker spellChecker;
}
```

By xml config, there is
```xml
<bean id="spellChecker" class="org.example.TextEditor" />
```