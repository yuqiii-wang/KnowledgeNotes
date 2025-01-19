# Spring Performance

## Reactive Streams

*Reactive Streams* are specification for **asynchronous** operation to handle data streams in a reactive, **non-blocking** way, especially when the producer (data source) and the consumer (data processor) operate at different speeds.

* WebFlux vs. Spring MVC

WebFlux is a reactive programming framework introduced in Spring 5 as part of the Spring ecosystem.

Spring MVC is based on the traditional Servlet API and uses a blocking I/O model.

WebFlux is designed for non-blocking, reactive programming and is better suited for high-concurrency, low-latency scenarios.

### Reactive Stream Players

#### `Publisher<T>`

In Project Reactor, Mono and Flux are implementations of `Publisher`.

#### `Subscriber<T>`

A `Subscriber` consumes the data emitted by the `Publisher`.

It has four methods:

* `onSubscribe(Subscription s)`: Called when the subscriber subscribes to the publisher.
* `onNext(T item)`: Called when a new data item is emitted.
* `onError(Throwable t)`: Called when an error occurs.
* `onComplete()`: Called when the stream completes successfully.

### Reactor: `Flux` and `Mono`

#### `Mono`

A `Mono` is a Publisher that emits **zero or one** item.

#### `Flux`

A `Flux` is a Publisher that emits **zero or more** item.

In the example below, `Flux.range(1, 100)` is a publisher to output 100 numbers.
This publisher has a callback defined in `.subscribe(...)`.

```java
Flux.range(1, 100) // Emits numbers from 1 to 100
    .subscribe(new Subscriber<Integer>() {
        private Subscription subscription;
        private int count = 0;

        @Override
        public void onSubscribe(Subscription s) {
            this.subscription = s;
            subscription.request(10); // Request the first 10 items
        }

        @Override
        public void onNext(Integer item) {
            System.out.println("Received: " + item);
            count++;
            if (count % 10 == 0) {
                subscription.request(10); // Request the next 10 items
            }
        }

        @Override
        public void onError(Throwable t) {
            System.err.println("Error: " + t.getMessage());
        }

        @Override
        public void onComplete() {
            System.out.println("Stream completed!");
        }
    });
```

### Functional Programming: `RouterFunction` and `HandlerFunction`

In Spring WebFlux, functional programming is an alternative to the traditional annotation-based programming model (e.g., `@Controller`, `@RequestMapping`).

In the code example below, the traditional java Spring accepting requests is replaced with handlers by routes.

```java
@Bean
public RouterFunction<ServerResponse> routes() {
    return RouterFunctions.route(
            RequestPredicates.GET("/hello/{name}"),
            this::handleHello
    ).andRoute(
            RequestPredicates.GET("/goodbye"),
            this::handleGoodbye
    );
}

private Mono<ServerResponse> handleHello(ServerRequest request) {
    String name = request.pathVariable("name");
    return ServerResponse.ok().bodyValue("Hello, " + name + "!");
}

private Mono<ServerResponse> handleGoodbye(ServerRequest request) {
    return ServerResponse.ok().bodyValue("Goodbye!");
}
```
