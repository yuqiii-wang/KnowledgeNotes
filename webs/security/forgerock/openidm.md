# OpenIDM

IDM provides a default schema for typical managed object types, such as users and roles (user refers to an execution entity, such as manager and employee, while role refers to associated execution permissions such as reviewing employee's performance by manager).

OpenIDM implements infrastructure modules that run in an OSGi (Open Services Gateway initiative) framework. It exposes core services through RESTful APIs to client applications.

## Core Services

* `Object Model`s are structures serialized or deserialized to and from JSON as required by IDM. 

* `Managed Object` is an object that represents the identity-related data managed by IDM. Managed objects are stored in the IDM repository. All managed objects are JSON-based data structures.

* `Mappings` define policies between source and target objects and their attributes during synchronization and reconciliation

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