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

Route requests to different handlers, this means

* handlers might be springboot projects hosted on different machine servers, that gateway acts as proxy managing IP-routing table by which it delegates requests
* handlers might have limited processing capability, that gateway acts as a loadbalancer assigning requests to idle handlers, and queues requests if all handlers are busy

For example, given `src/main/resources/application-routes-${ENV}.yml`

```yml
spring:
  cloud:
    gateway:
      routes:
        - id: example001.internal.com
          uri: http://example.internal.com
          predicates:
            - Path=/example001/**
          filters:
            - StringPrefix=1
            - name: Retry
              args:
                retries: 6
          metadata:
            response-timeout: 60000
            connect-timeout: 3000
        - id: example002.internal.com
          uri: http://another.com
          predicates:
            - Path=/example002/**
          filters:
            - StringPrefix=1
            - name: Retry
              args:
                retries: 6
          metadata:
            response-timeout: 60000
            connect-timeout: 3000
        - id: example003.internal.com
          uri: http://another.com
          predicates:
            - Path=/example003/**
          filters:
            - StringPrefix=1
            - name: Retry
              args:
                retries: 6
          metadata:
            response-timeout: 60000
            connect-timeout: 3000
```

The above handlers are registered by the below config.

`Flux<RouteDefinition> routeDefinitions` is an async publisher that `routeDefinitions.collectList().subscribe(...);` on finish of obtaining all route definitions (`.collectList()` is async awaiting the publisher `routeDefinitions` to return),
the `subscribe(propertiesRouteList::addAll)` callbacks `propertiesRouteList::addAll` so that `propertiesRouteList` is filled with obtained all route definitions.

Then `propertiesRouteList.stream().forEach(...)` starts running and loads predicates into `predicatePathSet`.

```java
package com.internal.example.gateway.config;

@Configuration
public class RouteDefinitionConfiguration {
    @Autowire
    private PropertiesRouteDefinitionLocator propertiesRouteDefinitionLocator; 

    private List<RouteDefinition> propertiesRouteList = new ArrayList<>();

    private Set<String> predicatePathSet = new HashSet<>();

    @PostConstruct
    private void postConstruct() {
        Flux<RouteDefinition> routeDefinitions = propertiesRouteDefinitionLocator.getRouteDefinitions();
        routeDefinitions.collectList().subscribe(propertiesRouteList::addAll);
        propertiesRouteList.stream().forEach(route -> {
            List<PredicateDefinition> predicates = route.getPredicates();
            if (!CollectionUtils.isEmpty(predicates)) {
                predicates.stream()
                    .filter(predicate -> predicate.getName().equalIgnoreCase("path"))
                    .forEach(predicate -> {
                        Map<String, String> args = predicate.getArgs();
                        Collection<String> pathValues = args.values();
                        if (!CollectionUtils.isEmpty(pathValues)) {
                            predicatePathSet.addAll(pathValues);
                        }
                    })
            }
        })
    }

    public Set<String> getPredicatePathSet() {
        return new HashSet<>(predicatePathSet);
    }
}
```

### IP Whitelist

In file `src/main/resources/application-whitelist-${ENV}.yml`,

```yml
whitelist:
    host:
        - example001.internal.com
        - example002.internal.com
        - example003.internal.com
```

In `src/main/java/com/internal/example/gateway/config/WhiteListConfig.java`

```java
package com.internal.example.gateway.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
@ConfigurationProperties(prefix="whitelist")
@Data
public class WhiteListConfig {
    private List<String> hosts;
}
```

Implement it:

```java
package com.internal.example.gateway.config;

import org.springframework.security.authentication.ReactiveAuthenticationManager;
import org.springframework.stereotype.Component;


@Component
public class ExampleAuthenticationManager implements ReactiveAuthenticationManager {
    @Autowired
    private WhiteListConfig whitelistConfig;

    ... // Some methods 
}
```

### MTLS

## External API Authentication

### User Identity Auth

#### SSO

### Keepie

## API Documentation

### Swagger
