# Microservice

*Microservices* is an architectural style that structures an application as a collection of small, loosely coupled services, where ideally, individual business functions are unilaterally linked individual components that can be easily removed/added/replaced.
It is the opposite to traditional monolithic software architecture where components are coupled with each other to deliver a business function.

* Microservice vs Servlet

Independence: Microservices are independently deployable and scalable services. Servlets are usually part of a monolithic web application and are not independently deployable.

## Interview Question: How do you service your functions as microservices

1. Explain basic functions a project would service, and how different functions can be grouped together and some be separated.
2. Based on aforementioned services, propose communication between servicing APIs (e.g., HTTP request/response).
3. Further dive deep into the most critical APIs where request/response body be explained to provide detailed clarification of the services
4. For these critical services/APIs, explain solution to high availability, resilience and security.
5. Database is always a critical service, explain how backup would be arranged, and services' data be separated in DB for fast query.
6. Go back to an overview of the architecture, generally speaking, how ssl certificates would be managed, and how authentication would be handled.

In summary, answer should focus on critical service separation with explained high availability, resilience and security.