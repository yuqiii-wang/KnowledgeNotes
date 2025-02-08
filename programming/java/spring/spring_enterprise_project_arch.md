# Spring Enterprise Project Architecture

```java
@EnableAsync
@SpringBootApplication
@DependsOn({"springFacade"})
@EnableFeignClients
@EnableDiscoveryClient
@EnableJms
@EnableDynamicTp
@EnableScheduling
@EnableRetry
public class YourEnterpriseProject {
    public static void main(string[] args) throws Exception {
        SpringBootApplication app = new SpringBootApplication(YourEnterpriseProject.class);
        app.run("--spring.profiles.active="+System.getProperty("spring.profiles.active"));
    }
}
```

## Multi-Threading in Spring

`@Controller` method is executed in the same thread that received the HTTP request (by default) from a servlet container, e.g., Tomcat.
Tomcat has a built-in thread pool, and if no other async executor specified, Tomcat's thread will go through the whole lifecycle of processing an HTTP request.

If the method is annotated with `@Async` (for asynchronous execution), or other thread pool for executor, it will be delegated to a separate thread from the configured thread pool.
If not specified, Tomcat and Spring share the same thread.

Be aware that thread pool is established per JVM, so that thread pool is **isolated** to each instance of the Spring Boot application.
In other words, if Spring applications are launched in multiple server machines, each server machine has a thread pool (assumed each server machine launched one JVM for one Spring application), not like the whole cluster has one thread pool.

### DynamicTP

### Example Use Case: Bulk Trade Query

The requirement: define a function to query trade info by a list of trade IDs.
However, out of data privacy regulation, trades' data is stored in different DB instances in different country locations, hence required to query by trade ID one by one.
To facilitate query, use multi-threading.

```java
public List<TradeInfo> fetchTradeInfo(List<String> tradeIds);
```

First, define a thread pool.

```java
import org.dromara.dynamictp.core.support.DynamicTp;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

@Configuration
@EnabledAsync
public class ThreadExecutorConfig {
    @Value("${async.executor.thread.core_pool_size}")
    private int corePoolSize;
    @Value("${async.executor.thread.max_pool_size}")
    private int maxPoolSize;

    @DynamicTp("taskExecutor")
    @Bean(destroyMethod="shutdown")
    public ThreadPoolTaskExecutor taskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(corePoolSize);
        executor.setMaxPoolSize(maxPoolSize);
        executor.setWaitForTasksToCompleteOnShutdown(true);
        return executor;
    }
}
```

For each trade id in `tradeIds.forEach`, query by `taskExecutor.execute`.

```java
@Service
public class TradeQueryService implements Serializable {
    @Qualifier("taskExecutor")
    @Autowired
    private ThreadPoolTaskExecutor taskExecutor;

    @Autowired
    private MySqlDBWrapper dbChinaTrade;
    @Autowired
    private MySqlDBWrapper dbUKTrade;
    @Autowired
    private MySqlDBWrapper dbUSTrade;
    @Autowired
    private MySqlDBWrapper dbJapanTrade;

    public List<TradeInfo> fetchTradeInfo(List<String> tradeIds) {
        List<TradeInfo> tradeInfoList = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch latch = new CountDownLatch(tradeIds.size());

        tradeIds.forEach(tradeId -> {
            taskExecutor.execute(() -> {
                try {
                    // Simulate fetching trade info for the given trade ID
                    TradeInfo tradeInfo = queryTradeInfo(tradeId);
                    tradeInfoList.add(tradeInfo);
                } finally {
                    latch.countDown();
                }
            });
        });

        try {
            latch.await(); // Wait for all tasks to complete
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Tasks interrupted", e);
        }

        return tradeInfoList;
    }

    private TradeInfo queryTradeInfo(String tradeId) {
        // Simulate a trade info query (e.g., database or API call)
        TradeInfo tradeInfo;
        if (tradeInfo == null) {
            tradeInfo = dbChinaTrade.getTradeInfo(tradeId);
        }
        if (tradeInfo == null) {
            tradeInfo = dbUKTrade.getTradeInfo(tradeId);
        }
        if (tradeInfo == null) {
            tradeInfo = dbUSTrade.getTradeInfo(tradeId);
        }
        if (tradeInfo == null) {
            tradeInfo = dbJapanTrade.getTradeInfo(tradeId);
        }
        if (tradeInfo == null) { // default empty tradeInfo
            tradeInfo = new TradeInfo(tradeId);
        }
        return tradeInfo;
    }
}
```

where

* `Collections.synchronizedList` ensures atomicity adding new elements
* `CountDownLatch` ensures all trade IDs returned, that every trade ID triggers `latch.countDown();`

## Multi Instance Cluster

### Example Cluster Distributed Lock

```xml
<dependency>
    <groupId>org.redisson</groupId>
    <artifactId>redisson</artifactId>
    <version>3.20.0</version>
</dependency>
```

```java
import org.redisson.Redisson;
import org.redisson.api.RLock;
import org.redisson.api.RedissonClient;
import org.redisson.config.Config;

import java.util.concurrent.TimeUnit;

public class RedisDistributedLockExample {
    public static void main(String[] args) {
        // Configure Redisson client
        Config config = new Config();
        config.useSingleServer()
                .setAddress("redis://127.0.0.1:6379") // Redis server address
                .setPassword(null); // Set password if needed

        // Create Redisson client
        RedissonClient redissonClient = Redisson.create(config);

        // Get a distributed lock
        RLock lock = redissonClient.getLock("my-distributed-lock");

        // Thread 1: Attempting to acquire the lock
        Thread thread1 = new Thread(() -> {
            try {
                if (lock.tryLock(5, 10, TimeUnit.SECONDS)) { // Wait 5 seconds, lock for 10 seconds
                    try {
                        System.out.println("Thread 1 acquired the lock");
                        Thread.sleep(3000); // Simulate some work
                    } finally {
                        lock.unlock();
                        System.out.println("Thread 1 released the lock");
                    }
                } else {
                    System.out.println("Thread 1 could not acquire the lock");
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });

        // Thread 2: Attempting to acquire the lock
        Thread thread2 = new Thread(() -> {
            try {
                if (lock.tryLock(5, 10, TimeUnit.SECONDS)) { // Wait 5 seconds, lock for 10 seconds
                    try {
                        System.out.println("Thread 2 acquired the lock");
                        Thread.sleep(3000); // Simulate some work
                    } finally {
                        lock.unlock();
                        System.out.println("Thread 2 released the lock");
                    }
                } else {
                    System.out.println("Thread 2 could not acquire the lock");
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });

        // Start threads
        thread1.start();
        thread2.start();

        // Shutdown Redisson after work
        Runtime.getRuntime().addShutdownHook(new Thread(redissonClient::shutdown));
    }
}
```

#### Redis Key Partition and Leader Node Election

## External Request by Feign Client

## Design Pattern: Facade

## Message Streaming

||WebSockets|MQ/JMS (Java Message Service)|
|-|-|-|
|Communication Type|Bidirectional, real-time, full-duplex|Asynchronous, message-based, store-and-forward|
|Protocol|WebSocket over TCP (`ws://`, `wss://`)|Messaging middleware (JMS API over TCP, AMQP, MQTT, etc.)|
|Architecture|Direct client-server connection|Producer-Consumer with a Message Broker|
|Latency|Very low (<1ms)|Higher due to message persistence and reliability|

### JMS

### Websocket

A java client can make websocket connection to `ChatWebSocketServer` by

```java
WebSocketContainer container = ContainerProvider.getWebSocketContainer();
            container.connectToServer(WebSocketClient.class, new URI("ws://localhost:8080/chat"));
```

The `ChatWebSocketServer` is defined as

```java
import jakarta.websocket.*;
import jakarta.websocket.server.ServerEndpoint;
import java.io.IOException;

@ServerEndpoint("/chat")
public class ChatWebSocketServer {

    @OnOpen
    public void onOpen(Session session) {
        System.out.println("Connected: " + session.getId());
    }

    @OnMessage
    public void onMessage(String message, Session session) throws IOException {
        System.out.println("Received: " + message);
        session.getBasicRemote().sendText("Echo: " + message);
    }

    @OnClose
    public void onClose(Session session) {
        System.out.println("Disconnected: " + session.getId());
    }

    @OnError
    public void onError(Session session, Throwable throwable) {
        throwable.printStackTrace();
    }
}
```

## Tracing, Monitoring and Analytics

### `prometheus` and Grafana for monitoring

### `sleuth` for tracing

### `zipkin` for data collection and analytics
