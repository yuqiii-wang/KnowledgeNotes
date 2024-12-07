# Tomcat

## File structure

* bin - startup, shutdown and other scripts and executables
* common - common classes that Catalina and web applications can use
* conf - XML files and related DTDs to configure Tomcat
* logs - Catalina and application logs
* server - classes used only by Catalina
* shared - classes shared by all web applications
* webapps - directory containing the web applications (copy your java applications to here)
* work - temporary storage for files and directories

### `web.xml` in `conf` vs in `WEB-INF`

* `web.xml` in `conf`

A deployment descriptor which is applied to the current web application only and as such controls the running of just that web app. It allows you define your servlets, servlet mapping to URLs, context (startup) parameters etc.

* `web.xml` in `WEB-INF`

It contains the minimum set of settings required to get your webapps to work properly, 
defining the default parameters for ALL applications on a Tomcat instance.

### WEB-INF and META-INF

By Servlet specifications, `WEB-INF` is used to store non-public static files, such as `.js` and config files. `META-INF` stores java classes.

## Servlets

A servlet is a java request-response programming model.

`Listener`s and `Filter`s are tomcat special type servlets. `Listener` monitors events and `Filter` rejects requests based on some rules, in which one typical is against CSRF and one forbidding non-SSL requests.

Tomcat start sequence:
1. ServletContext: Tomcat servlet init
2. listener
3. filter
4. servlet: user defined servlets

## Connector

Each Connector element represents a port that Tomcat will listen to for requests.  By arranging these Connector elements within hierarchies of Services and Engines, a Tomcat administrator is able to create a logical infrastructure for data to flow in and out of their site.  

For example, here defines two web apps listening to two ports.

```xml
<Server>

  <Service name="Catalina">
    <Connector port="8443"/>
    <Engine>
      <Host name="yourhostname">
        <Context path="/webapp1"/>
      </Host>
    </Engine>
  </Service>
 
  <Service name="Catalina8444">
    <Connector port="8444"/>
    <Engine>
      <Host name="yourhostname">
        <Context path="/webapp2"/>
      </Host>
    </Engine>
  </Service>

</Server>
```

Another example, `http-nio-6610` (often observed in tomcat/spring handling HTTP requests) says about

* `HTTP`: Specifies that this connector is for handling HTTP traffic.
* `NIO`: Indicates the use of the Non-blocking I/O model, which allows handling multiple client connections simultaneously with fewer threads. This is particularly efficient for high-concurrency scenarios.
* `6610`: This is the port number on which the connector listens for incoming HTTP requests.

```xml
<Connector port="6610" protocol="org.apache.coyote.http11.Http11NioProtocol"
           connectionTimeout="20000"
           redirectPort="8443" />
```

### Types of Connectors

* HTTP connectors

It's set by default to HTTP/1.1.

Setting the "SSLEnabled" attribute to "true" causes the connector to use SSL handshake/encryption/decryption.  

Used as part of a load balancing scheme and proxy.

* AJP connectors

Apache JServ Protocol, or AJP, is an optimized binary version of HTTP that is typically used to allow Tomcat to communicate with an Apache web server.

## Configs

### Server.xml

The elements of the `server.xml` file belong to five basic categories - Top Level Elements, Connectors, Containers, Nested Components, and Global Settings.

The port attribute of `Server` element is used to specify which port Tomcat should listen to for shutdown commands.

`Service` is used to contain one or multiple Connector components that share the same Engine component.

By nesting one `Connector` (or multiple Connectors) within a Service tag, you allow Catalina to forward requests from these ports to a single Engine component for processing.

`Listener` can be nested inside Server, Engine, Host, or Context elements, point to a component that will perform an action when a specific event occurs. Two most typical are listeners to startup and shutdown signals.

`Resource` directs Catalina to static resources used by your web applications.

#### Example

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Server port="8005" shutdown="SHUTDOWN">
    <!-- Global JNDI Resources -->
    <GlobalNamingResources>
        <Resource name="UserDatabase" auth="Container"
                  type="org.apache.catalina.UserDatabase"
                  description="User database that can be updated and saved"
                  factory="org.apache.catalina.users.MemoryUserDatabaseFactory"
                  pathname="conf/tomcat-users.xml" />
    </GlobalNamingResources>

    <Service name="Catalina">
        <!-- Define the HTTP connector using the NIO protocol -->
        <Connector port="6610" 
                   protocol="org.apache.coyote.http11.Http11NioProtocol"
                   connectionTimeout="20000"
                   redirectPort="8443"
                   maxThreads="200"
                   minSpareThreads="10"
                   acceptCount="100"
                   enableLookups="false"
                   URIEncoding="UTF-8" />

        <!-- Define the AJP connector for backend integration (optional) -->
        <!-- <Connector port="8009" protocol="AJP/1.3" redirectPort="8443" /> -->

        <!-- Engine, Host, and Context configuration -->
        <Engine name="Catalina" defaultHost="localhost">
            <Realm className="org.apache.catalina.realm.LockOutRealm">
                <Realm className="org.apache.catalina.realm.UserDatabaseRealm"
                       resourceName="UserDatabase" />
            </Realm>
            <Host name="localhost" appBase="webapps"
                  unpackWARs="true" autoDeploy="true">
                <Valve className="org.apache.catalina.valves.AccessLogValve"
                       directory="logs" prefix="localhost_access_log" suffix=".txt"
                       pattern="%h %l %u %t &quot;%r&quot; %s %b" />
            </Host>
        </Engine>
    </Service>
</Server>
```

### Web.XML in `conf`

Tomcat will use TOMCAT-HOME/conf/web.xml as a base configuration, which can be overwritten by application-specific `WEB-INF/web.xml` files.

### Web.XML in `webapps/conf`

It provides app-customized configs.