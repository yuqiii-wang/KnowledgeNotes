# OpenIDM

IDM provides a default schema for typical managed object types, such as users and roles (user refers to an execution entity, such as manager and employee, while role refers to associated execution permissions such as reviewing employee's performance by manager).

OpenIDM implements infrastructure modules that run in an OSGi (Open Services Gateway initiative) framework. It exposes core services through RESTful APIs to client applications.

## Core Services

* `Object Model`s are structures serialized or deserialized to and from JSON as required by IDM. 

* `Managed Object` is an object that represents the identity-related data managed by IDM. Managed objects are stored in the IDM repository. All managed objects are JSON-based data structures.

* `Mappings` define policies between source and target objects and their attributes during synchronization and reconciliation

P.S. Object management service is similar in terms of service concept to Java Hibernate that performs object mappings between Java objects and relational db storage.

## DB Connection

To store object data to a repository (here we use mysql db), main steps are:

Reference:
https://backstage.forgerock.com/docs/idm/7/install-guide/repository-mysql.html

1. `MySQL Connector` (download it) is an ODBC interface and allows programming languages that support the ODBC interface to communicate with a MySQL database. 

2. copy mysql client/connector config files to openidm project folders:
```bash
cp /path/to/openidm/db/mysql/conf/datasource.jdbc-default.json my-project/conf/
cp /path/to/openidm/db/mysql/conf/repo.jdbc.json my-project/conf/
```

3. load openidm sql

Such sql scripts map openidm defined data to mysql-compliant storage. 
```bash
mysql -u root -p < /path/to/openidm/db/mysql/scripts/openidm.sql
mysql -u root -p < /path/to/openidm/db/mysql/scripts/createuser.sql
mysql -D openidm -u root -p < /path/to/openidm/db/mysql/scripts/flowable.mysql.all.create.sql
```
For example, inside repo.jdbc.json
```sql
SELECT objectid FROM ${_dbSchema}.${_table} LIMIT ${int:_pageSize} OFFSET ${int:_pageResultOffset}
```

4. update `jdbcUrl` on `conf/datasource.jdbc-default.json`
```json
{"jdbcUrl" : "jdbc:mysql://&{openidm.repo.host}:&{openidm.repo.port}/openidm?allowMultiQueries=true&characterEncoding=utf8&serverTimezone=UTC"}
```
through specifying values on `resolver/boot.properties`
```bash
openidm.repo.host=localhost
openidm.repo.port=3306
```

### Config Files

* The `boot.properties` file is the property resolver file. Generally, this file lets you set variables that are used in other configuration files.

*  The `config.properties` file is used for two purposes:

    1) To set OSGi bundle properties.

    2) To set Apache Felix properties, and plugin bundles related to the Felix web console. 

* The `system.properties` file is used to bootstrap java system properties such as:

    1) Jetty log settings, based on the Jetty container bundled with IDM. IDM bundles Jetty version 9.4.22.

    2) Cluster configuration.

    3) Quartz updates, as described in Quartz Best Practices documentation.

    4) A common transaction ID, as described in "Configure the Audit Service". 

## Some Checks

* Health Check
```bash
curl \
--header "X-OpenIDM-Username: openidm-admin" \
--header "X-OpenIDM-Password: openidm-admin" \
--header "Accept-API-Version: resource=1.0" \
--request GET \
"http://localhost:8080/openidm/info/ping"
```

* current IDM session
```bash
curl \
--header "X-OpenIDM-Username: openidm-admin" \
--header "X-OpenIDM-Password: openidm-admin" \
--header "Accept-API-Version: resource=1.0" \
--request GET \
"http://localhost:8080/openidm/info/login"
```

* create an ldap user
```bash
curl \
 --header "Content-Type: application/json" \
 --header "X-OpenIDM-Username: openidm-admin" \
 --header "X-OpenIDM-Password: openidm-admin" \
 --header "Accept-API-Version: resource=1.0" \
 --request POST \
 --data '{
 "dn": "CN=Brian Smith,CN=Users,DC=example,DC=com",
 "cn": "Brian Smith",
 "sAMAccountName": "bsmith",
 "userPrincipalName": "bsmith@example.com",
 "userAccountControl": "512",
 "givenName": "Brian",
 "mail": "bsmith@example.com",
 "__PASSWORD__": "Passw0rd"
 }' \
 http://localhost:8080/openidm/system/ad/account?_action=create
```

## Reconciliation

Reconciliation is a practice that compares diffs between two or more data stores and checking for missing data, then performs synchronization of between data stores.

* To initiate reconciliation
```bash
curl \
 --cacert self-signed.crt \
 --header "X-OpenIDM-Username: openidm-admin" \
 --header "X-OpenIDM-Password: openidm-admin" \
 --request POST \
 "https://localhost:8443/openidm/recon?_action=recon&mapping=systemLdapAccounts_managedUser"
```
Here, the name of the mapping `systemLdapAccounts_managedUser`, is defined in the `conf/sync.json` file.

Example `conf/sync.json`
```json
{
    "mappings": [
        {
            "name": "managedUser_systemLdapAccounts",
            "source": "managed/user",
            "target": "system/MyLDAP/account",
            "linkQualifiers" : {
                "type" : "text/javascript",
                "file" : "script/linkQualifiers.js"
            }
        }
    ]
}
```

* To list reconciliation runs
```bash
curl \
 --cacert self-signed.crt \
 --header "X-OpenIDM-Username: openidm-admin" \
 --header "X-OpenIDM-Password: openidm-admin" \
 --request GET \
 "https://localhost:8443/openidm/recon"
```