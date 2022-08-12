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
definING the default parameters for ALL applications on a Tomcat instance.

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

### Web.XML

Tomcat will use TOMCAT-HOME/conf/web.xml as a base configuration, which can be overwritten by application-specific `WEB-INF/web.xml` files.