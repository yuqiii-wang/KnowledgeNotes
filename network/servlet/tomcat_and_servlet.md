# Tomcat and Servlet

Services in tomcat by servlets are shown below.

![servlets_in_tomcat](imgs/servlets_in_tomcat.png "servlets_in_tomcat")

`Service` is the outer layer interface receiving requests, such as opening a socket listening a port

`Host`, such as `example.com`, represents a site; each host can have multiple `Context` contained/referenced in `WEB-INF/Web.xml`

`Context` is an app represented in `webapps`, configured by `conf/web.xml` under the `webapps` directory

`Connector` is used to process data from port.

`Executor` is used to maintain servlet threads.

`Engine` is used to config hosts (domain names)

Each `Wrapper` corresponds to one servlet.

## ServletContext

ServletContext is responsible for servlet lifecycle management.

On every HTTP request, ServletContext retrieves one thread from a thread pool. After the service finished, it returns to the thread pool.

ServletContext can handle decoding data, such as a customized protocol.

## Servlet Definition

```java
public interface Servlet {

 public void init(ServletConfig config) throws ServletException;

 public ServletConfig getServletConfig();
 // handle user's request
 public void service(ServletRequest req, ServletResponse res)
    throws ServletException, IOException;
 // return such as code owner and copyright
 public String getServletInfo();

 public void destroy();
}
```
where
```java
public interface ServletConfig {

 // find servlets from web.xml
 public String getServletName();
 
 // ServletContext is global, against others being local to a particular servlet
 public ServletContext getServletContext();
 
 // get init-param from web.xml
 public String getInitParameter(String name);
 
 public Enumeration getInitParameterNames();
}
```

`HttpServlet` defines HTTP methods.
```java
public abstract class HttpServlet extends GenericServlet implements java.io.Serializable {
	private static final String METHOD_DELETE = "DELETE";
	private static final String METHOD_HEAD = "HEAD";
	private static final String METHOD_GET = "GET";
	private static final String METHOD_OPTIONS = "OPTIONS";
	private static final String METHOD_POST = "POST";
	private static final String METHOD_PUT = "PUT";
	private static final String METHOD_TRACE = "TRACE";

	private static final String HEADER_IFMODSINCE = "If-Modified-Since";
	private static final String HEADER_LASTMOD = "Last-Modified";

	private static final String LSTRING_FILE = "javax.servlet.http.LocalStrings";
	private static ResourceBundle lStrings = ResourceBundle.getBundle(LSTRING_FILE);

	/**
	 * Does nothing, because this is an abstract class. 抽象类 HttpServlet
	 */
	public HttpServlet() {
	}

    // Many method implementations

    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        String protocol = req.getProtocol();
        String msg = lStrings.getString("http.method_get_not_supported");
        if (protocol.endsWith("1.1")) {
            resp.sendError(HttpServletResponse.SC_METHOD_NOT_ALLOWED, msg);
        } else {
            resp.sendError(HttpServletResponse.SC_BAD_REQUEST, msg);
        }
    }

    protected long getLastModified(HttpServletRequest req) {
        return -1;
    }

    protected void doHead(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        NoBodyResponse response = new NoBodyResponse(resp);

        doGet(req, response);
        response.setContentLength();
    }

    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        String protocol = req.getProtocol();
        String msg = lStrings.getString("http.method_post_not_supported");
        if (protocol.endsWith("1.1")) {
            resp.sendError(HttpServletResponse.SC_METHOD_NOT_ALLOWED, msg);
        } else {
            resp.sendError(HttpServletResponse.SC_BAD_REQUEST, msg);
        }
    }

    protected void doPut(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        String protocol = req.getProtocol();
        String msg = lStrings.getString("http.method_put_not_supported");
        if (protocol.endsWith("1.1")) {
            resp.sendError(HttpServletResponse.SC_METHOD_NOT_ALLOWED, msg);
        } else {
            resp.sendError(HttpServletResponse.SC_BAD_REQUEST, msg);
        }
    }

    protected void doDelete(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        String protocol = req.getProtocol();
        String msg = lStrings.getString("http.method_delete_not_supported");
        if (protocol.endsWith("1.1")) {
            resp.sendError(HttpServletResponse.SC_METHOD_NOT_ALLOWED, msg);
        } else {
            resp.sendError(HttpServletResponse.SC_BAD_REQUEST, msg);
        }
    }

    ...
}
```

![request_to_servlet](imgs/request_to_servlet.png "request_to_servlet")
