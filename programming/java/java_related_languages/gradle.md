# Gradle

Gradle is an advanced build automation tool.
Gradle is the successor of Maven, and can be used in c++, python, javascript projects.

## Core Concepts: Gradle vs. Maven

|Concept|Gradle|Maven Equivalent|
|-|-|-|
|Build Script|`build.gradle.kts` or `build.gradle`: A programmable script using Kotlin or Groovy.|`pom.xml` (Project Object Model). A declarative XML file that describes the project and its configuration.|
|Unit of Work|Task A single action, like compileJava or test.|Goal - the lifecycle consists of phases (e.g., validate, compile, test, package)|

## Multi-Module Large Gradle Project

A typical web application might be structured like this:

* my-app-api: Contains just the data transfer objects (DTOs) and interface definitions shared between the client and server.
* my-app-core: Contains the core business logic, services, and domain models.
* my-app-data: Contains the database repository layer (e.g., using JPA/Hibernate).
* my-app-web: Contains the REST controllers and web layer (e.g., using Spring MVC).

Gradle action:

1. The `settings.gradle.kts` file is the first thing Gradle reads. Its primary job is to define the structure of your multi-module project.
2. The `build.gradle.kts` file in project root directory is used to define configuration that is common to all the sub-projects.
3. The Sub-project `build.gradle.kts` defines the configuration, dependencies, and tasks that are unique to that specific module.