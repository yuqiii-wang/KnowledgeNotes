# System Design

## Principles:

* Reliability: resilient in terms of fault tolerance, single point of failure

* Availability: performance during peak service time, average delay in response

* Scalability: easy in terms of scaling up/down when service demands change

* Sustainability: easy to maintain, components mostly separable 

* Security: connections between different components should be secure and only have minimum right of access (e.g., internal certificate management, different level API exposure scenarios)

## Typical Project Flow

1. Define Use Cases/Requirements, 

Commonly, there are registration, login, preference settings, etc.

2. Draw UML gragh for user journey, explain relationships/behavior of different entities.

Vaguely define attributes of entities and methods how entities interact with each other.

3. Technical system architecture design.

Abstract business requirements into technical implementations. Technical considerations vary according to business requriements, such as demand (concurrency), volume, budget.

4. Agile development

Decompose the whole project into different phases, start with core functionalities as phase/day ZERO, and proceed. 

Always engage clients/customers to see if delivery of each phase is satisfactory to their needs, and be agile/adaptable to next development.

## Critical points of consideration

* Concurrency

* Security

Internal APIs should not be exposed to the internet.

* Single point of failure

* DB consumption

DB clustering

DB concurrency

DB Indexing: Element/data indexing with proper business requriement understanding (what fields often got searched by) 

DB I/O cost.

* Budget 

Combine/separate development envs. For example, DB is often expensive, on dev env, there can be only one db cluster with different db host for dev, qa, preprod envs; serverless services (AWS Lambda) cost little, hence separable.