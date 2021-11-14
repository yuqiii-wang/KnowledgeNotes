# Tomcat

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

`Listener` can be nested inside Server, Engine, Host, or Context elements, point to a component that will perform an action when a specific event occurs.

`Resource` directs Catalina to static resources used by your web applications.

### Web.XML

Tomcat will use TOMCAT-HOME/conf/web.xml as a base configuration, which can be overwritten by application-specific `WEB-INF/web.xml` files.