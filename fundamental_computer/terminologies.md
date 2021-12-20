# Terminology Explain

* Concurrency & Parallelism

`Concurrency` means executing multiple tasks at the same time but not simultaneously. E.g. two tasks works in overlapping time periods.

`Parallelism` means performing two or more tasks simultaneously, e.g. performing multiple calculations at the same time.

* Serialization

In computing, serialization is the process of translating a data structure or object state into a format that can be stored (for example, in a file or memory data buffer) or transmitted (for example, over a computer network) and reconstructed later (possibly in a different computer environment).

This process of serializing an object is also called `marshalling` an object in some situations.

For example, `Go` natively supports unmarshalling/marshalling of JSON and XML data, while Java provides automatic serialization which requires that the object be marked by implementing the java.io.Serializable interface.


* Servlet

A code snippet running on a server. Every HTTP request is sent and processed in a web container. Business user sends requests from browser (applet, applet container), through HTTP SSL a web container handles the request. A servlet consists of a number of components, such as object instantiations when receiving a request, and garbage collection after a complete HTTP request/response finishes. After, an EJB container runs that provides multi-threading execution.

A web container can be regarded as a special JVM tool interface that manages the servlets and a thread pool. One example is that a JSP page is translated between HTML and java code.

* ABI (application binary interface)

In general, an ABI is the interface between two program modules, one of which is often at the level of machine code. The interface is the de facto method for encoding/decoding data into/out of the machine code.

In Ethereum, it's basically how you can encode Solidity contract calls for the EVM and, backwards, how to read the data out of transactions.