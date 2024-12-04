# API Hosting For External Use

Some config for load balancer, firewall, service discovery, authentication, etc., for API be exposed for external use.

## Service Discovery

Spring Cloud provides *Service Discovery* that registers what APIs be bound to what services on what servers/clients, as well as API documentation and registry.

### Eureka

* Developed by Netflix as part of the Netflix OSS (Open Source Software) stack.
* Written in Java.
* No built-in Key-Value Store

### Consul

* Developed by HashiCorp.
* Written in Go.
* Key-Value Store: A hierarchical key-value store for dynamic configuration.

## Firewall and Gateway

### Routing

### IP Whitelist

### MTLS

## External API Authentication

### User Identity Auth

#### SSO

### Keepie

## API Documentation

### Swagger
