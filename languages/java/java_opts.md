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