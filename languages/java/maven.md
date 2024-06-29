# Maven

## .m2 Config and `~/.m2/setting.xml`

Make sure `M2_HOME` (for maven repository) set properly for Maven

For a Maven to use CN Mainland mirrors, add the following in Maven root dir `~/.m2/settings.xml`

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

### Setting Security

#### Security Password

First, run `mvn --encrypt-password <user-password>` that outputs `{encrypted_master_password}`.
In `${USER_HOME}/.m2/settings-security.xml`, copy `{encrypted_master_password}` into the below.

```xml
<settingsSecurity>
    <master>{encrypted_master_password}</master>
</settingsSecurity>
```

Run `mvn --encrypt-password <user-password>` that outputs `{encrypted_password_password}`.
In `${USER_HOME}/.m2/settings.xml`, copy `{encrypted_password_password}` into the below.

```xml
<settings>
    <mirrors>
        <mirror>
            <id>example-mirror-id</id>
            <mirrorOf>central</mirrorOf>
            <url>https://your-mirror-repository-url</url>
        </mirror>
    </mirrors>
    <servers>
        <server>
            <id>example-mirror-id</id>
            <username>your-username</username>
            <password>{encrypted_password_password}</password>
        </server>
    </servers>
</settings>
```

#### Certificate

Reference:

https://stackoverflow.com/questions/21076179/pkix-path-building-failed-and-unable-to-find-valid-certification-path-to-requ

https://www.cnblogs.com/wpbxin/p/11746229.html

Check from browser that the repo is permitted to access.

If permitted, from browser export the cert as `example-mirror-id.cer`.

<div style="display: flex; justify-content: center;">
    <img src="imgs/cert_from_chrome.png" width="30%" height="40%" alt="cert_from_chrome" />
</div>
</br>

Add `example-mirror-id.cer` to JRE security.

Failed to complete this step might raise "PKIX path building failed: sun.security.provider.certpath.SunCertPathBuilde" error.

```sh
keytool -import -alias example-mirror -keystore  "/path/to/<jre-version>/lib/security/cacerts" -file example-mirror-id.cer
```

## pom.xml

A minimal `POM.xml`

```xml
<project>
  <modelVersion>4.0.0</modelVersion>
 
  <groupId>com.mycompany.app</groupId>
  <artifactId>my-app</artifactId>
  <version>1</version>
</project>
```

## Maven Setup in IntelliJ IDEA

1. Make sure Maven plugin is installed.

<div style="display: flex; justify-content: center;">
    <img src="imgs/idea_maven_plugin_is_install.png" width="50%" height="50%" alt="idea_maven_plugin_is_install" />
</div>
</br>

2. Add cert so that IDEA can trust a maven repository host to download dependencies.

<div style="display: flex; justify-content: center;">
    <img src="imgs/idea_trust_certs.png" width="50%" height="50%" alt="idea_trust_certs" />
</div>
</br>

1. Add custom Maven settings, where `Override` should be ticked to force using custom settings.

<div style="display: flex; justify-content: center;">
    <img src="imgs/idea_maven_seetings.png" width="50%" height="50%" alt="idea_maven_seetings" />
</div>
</br>

4. Index/sync between remote repo vs local env by `Update`, otherwise maven may see errors about many dependencies not found, though repos are present in maven repo host.

<div style="display: flex; justify-content: center;">
    <img src="imgs/idea_maven_repo_indexing.png" width="50%" height="50%" alt="idea_maven_repo_indexing" />
</div>
</br>

5. If Maven UI tab is not seen from the IDEA, make sure there is a `pom.xml` file present in the project, then add the project as a maven project.

<div style="display: flex; justify-content: center;">
    <img src="imgs/idea_add_as_a_maven_project.png" width="20%" height="90%" alt="idea_add_as_a_maven_project" />
</div>
</br>

6. Having all set up, trigger Maven for downloading dependencies by `refresh`, then select `maven clean` to remove cached builds, and `maven package` to compile the project.

<div style="display: flex; justify-content: center;">
    <img src="imgs/idea_maven_download_deps_and_build.png" width="70%" height="40%" alt="idea_maven_download_deps_and_build" />
</div>
</br>
