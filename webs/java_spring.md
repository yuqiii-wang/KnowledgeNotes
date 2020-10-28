# Java Spring and Spring Boot

* Some Starter Tips

Make sure `M2_HOME` (for maven repository) set properly for Maven

For a Maven to use CN Mainland mirrors, add the following in Maven root dir `conf/setting.xml`
```xml
<mirror>
   <id>alimaven</id>
   <name>aliyun maven</name>
　　<url>http://maven.aliyun.com/nexus/content/groups/public/</url>
   <mirrorOf>central</mirrorOf>        
</mirror>
```

Change encoding to UTF-8

* Dependency Injection

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