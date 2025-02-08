# Groovy Basics

Groovy is an object oriented language which is based on Java platform. 

It supports both static and dynamic typing, and added syntactic sugar makes it easy for scripting.

### Groovy Closure 

A closure is a short anonymous block of code. For example,
```groovy
class Example {
   static void main(String[] args) {
      def clos = {println "Hello World"};
      clos.call();
   } 
}
```

### Domain Specific Language (DSL)

Groovy allows one to omit parentheses around the arguments of a method call for top-level statements. This is known as the "command chain" feature. For example, 

```groovy
class EmailDsl {
    String toText 
    String fromText 
    String body 

    def static make(closure) {
        EmailDsl emailDsl = new EmailDsl()
        closure.delegate = emailDsl
        closure()
    }

    def to(String toText) { 
       this.toText = toText 
    }
   
    def from(String fromText) { 
      this.fromText = fromText 
    }
   
    def body(String bodyText) { 
       this.body = bodyText 
    } 
}

EmailDsl.make { 
   to "Xi Jinping" 
   from "Barack Obama" 
   body "Morning president."
}
```
and the execution result:
```bash
Morning president.
```

## Jenkins Pipeline by Groovy Domain Specific Language (DSL)

A Jenkinsfile is a text file that contains the definition of a Jenkins Pipeline and is checked into source control. In the context of groovy's DSL implementation, a Jenkinsfile is a Declarative Pipeline following the same rules as groovy's syntax with the following exceptions:

* The top-level of the Pipeline must be a block, specifically: `pipeline { }`.
* No semicolons as statement separators. Each statement has to be on its own line.
* Blocks must only consist of Sections, Directives, Steps, or assignment statements.
* A property reference statement is treated as a no-argument method invocation. So, for example, input is treated as input().

