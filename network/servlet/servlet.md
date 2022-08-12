# Servlet

## Intro

A servlet is a Java technology-based Web component (java class), managed by a container (aka servlet engine, e.g., tomcat), that generates dynamic content, implementing `javax.servlet.Servlet` interface.

For detailed servlet specification, please refer to Oracle Java website.

## Life Cycle

1. Loading and Instantiation

Example: `systemctl restart tomcat` that loads and instantiates web apps.

2. Initialization

Example: reading many tomcat's xml files to config the servlet container.

3. Request Handling

## Example:

A servlet container (e.g., tomcat) receives an https request. It processes tls and builds a connection, as well as services including bandwidth throttling, MIME data decoding, etc., then determines which servlet to invoke.

The invoked servlet loads the request and looks into what method, parameters and data the request contains. After processing logic provided the request, the servlet sends a response.

Finally, the servelet container makes sure the response is flushed and close the connection.

### Code example

Tomcat binds a particular request to an url to a Java Spring function.

In java spring
```java
@WebServlet("/view_book")
public class ViewBookServlet extends HttpServlet {
    ...
}
```

In Tomcat config
```xml
<servlet-mapping>
    <servlet-name>ViewBookServlet</servlet-name>
    <url-pattern>/view_book</url-pattern>
</servlet-mapping>
```

## Somr specifications

This specification document states the java method name that handles requests such as `doGet` for processing GET request, `doPost` for processing GET request.

reference:
https://download.oracle.com/otn-pub/jcp/servlet-2.4-fr-spec-oth-JSpec/servlet-2_4-fr-spec.pdf?AuthParam=1659722181_7ae40afa61c65c1aa2f1448c000e4623

`WEB-INF` folder's files are used by servlet java code but not visible to public.