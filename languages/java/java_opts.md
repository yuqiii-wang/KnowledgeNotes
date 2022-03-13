# Java Options

* `-Xms` and `-Xmx`

`Xmx` specifies max memory pool for JVM (Java Virtual Machine), `Xms` for initial allocated memory to jvm.

For example, 
```bash
-java -Xms256m -Xmx2048m
```

* `-server` and `-client`

JVM is tuned for either server or client services. 

`-server` JVM is optimize to provide best peak operating speed, executing long-term running applications, etc.

`-client` JVM provides great support for GUI applications, fast app startup time, etc.

* `-XX:+UseConcMarkSweepGC`

The Concurrent mark sweep collector (concurrent mark-sweep collector, concurrent collector or CMS) was a mark-sweep garbage collector in the Oracle HotSpot Java virtual machine (JVM) available since version 1.4.1., deprecated on version 9 and removed on version 14.

It is targeted at applications that are sensitive to garbage collection pauses. It performs most garbage collection activity concurrently, i.e., while the application threads are running, to keep garbage collection-induced pauses short. 