# Servlet

## Intro

A servlet is a Java technology-based Web component (java class), managed by a container (aka servlet engine, e.g., tomcat), that generates dynamic content, implementing `javax.servlet.Servlet` interface.

For detailed servlet specification, please refer to Oracle Java website.

### Life Cycle

1. Loading and Instantiation

Example: `systemctl restart tomcat` that loads and instantiates web apps.

2. Initialization

Example: reading many tomcat's xml files to config the servlet container.

3. Request Handling

Example:

A servlet container (e.g., tomcat) receives an https request. It processes tls and builds a connection, as well as services including bandwidth throttling, MIME data decoding, etc., then determines which servlet to invoke.

The invoked servlet loads the request and looks into what method, parameters and data the request contains. After processing logic provided the request, the servlet sends a response.

Finally, the servelet container makes sure the response is flushed and close the connection.