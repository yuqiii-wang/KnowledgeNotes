# Maven

## .m2 Config

Make sure `M2_HOME` (for maven repository) set properly for Maven

For a Maven to use CN Mainland mirrors, add the following in Maven root dir `~/.m2/setting.xml`
```xml
<mirror>
   <id>alimaven</id>
   <name>aliyun maven</name>
　　<url>http://maven.aliyun.com/nexus/content/groups/public/</url>
   <mirrorOf>central</mirrorOf>        
</mirror>
```

Change encoding to UTF-8

* `mvn clean`

This command cleans the maven project by deleting the target directory.

* `mvn compile`

This command compiles the java source classes of the maven project.

* `mvn package`

This command builds the maven project and packages them into a JAR, WAR, etc.

## pom.xml
