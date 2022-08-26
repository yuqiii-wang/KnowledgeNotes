# Complex System Naming Convention

## System API vs Process API

SAPI (System API) is built on system specific infras, where actual connection to databases occurs.

PAPI (Process API) is for logic and orchestration, where, for example, an "order" request is sent to PAPI for a product, and the info of the product (such as price, description, manufacture location, etc.) reside on different databases, then PAPI gathers various responses from SAPIs and forge a new semantic response to the "order" response.

![api-led-connectivity](imgs/api-led-connectivity.png "api-led-connectivity")
