# OpenAM

AM manages access to resources, such as a web page, an application, or a web service, that are available over the network by building a authentication flow of nodes.

AM handles both authentication and authorization, via such as LDAP (Lightweight Directory Access Protocol), Windows authentication, one-time password services.

## Policy

AM is an policy server that checks against client and user profile and manages tokens by defined policies.

### policy example

`Policy` can be used for taking action (analyzing requests) based on Opendj's user data (e.g., what permissions associated with a client id) to grant the requested permissions.

**reference:**
https://backstage.forgerock.com/knowledge/kb/book/b91932752#a10205600

1. use `amadmin` to obtain config permission (use response token)
```bash
curl -X POST \
    -H "X-OpenAM-Username: amadmin" \
    -H "X-OpenAM-Password: cangetinam" \
    -H "Content-Type: application/json" \
    -H "Accept-API-Version: resource=2.1" \
http://host1.example.com:8080/openam/json/realms/root/authenticate?authIndexType=service&authIndexValue=adminconsoleservice
```
It returns
```json
{ "tokenId": "AQIC5wM2LY4SfcxsuvGEjcsppDSFR8H8DYBSouTtz3m64PI.*AAJTSQACMDIAAlNLABQtNTQwMTU3NzgxODI0NzE3OTIwNAEwNDU2NjE0*", "successUrl": "/openam/console", "realm": "/" } 
```

2. find resource type ID; The "uuid" shown in the response is the "resourceTypeUuids" attribute required to create a policy.
```bash
token=AQIC5wM2LY4Sfcxs...EwNDU2NjE0*
curl \
    -H "iPlanetDirectoryPro: ${token}" \
http://host1.example.com:8080/openam/json/realms/root/resourcetypes?_queryFilter=true
```
It returns
```json
{
  "result": [
    {
      "uuid": "76656a38-5f8e-401b-83aa-4ccb74ce88d2",
      "name": "URL",
      "description": "The built-in URL Resource Type available to OpenAM Policies.",
      "patterns": [
        "*://*:*/*?*",
        "*://*:*/*"
      ],
      "actions": {
        "POST": true,
        "PATCH": true,
        "GET": true,
        "DELETE": true,
        "OPTIONS": true,
        "HEAD": true,
        "PUT": true
      },
      "createdBy": "id=dsameuser,ou=user,dc=openam,dc=forgerock,dc=org",
      "creationDate": 1422892465848,
      "lastModifiedBy": "id=dsameuser,ou=user,dc=openam,dc=forgerock,dc=org",
      "lastModifiedDate": 1422892465848
    }
  ]
}
```

3. update a policy according to resource uuid
```bash
curl -X POST \
-H "iPlanetDirectoryPro: AQIC5wM2LY4Sfcxs...EwNDU2NjE0*" \
-H "Content-Type: application/json" \
-H "Accept-API-Version: resource=2.0" \
-d "{
    "name": "mypolicy",
    "active": true,
    "description": "My Policy.",
    "applicationName": "iPlanetAMWebAgentService",
    "actionValues": {
        "POST": false,
        "GET": true
    },
    "resources": [
        "http://www.example.com:80/*",
        "http://www.example.com:80/*?*"
    ],
    "subject": {
        "type": "Identity",
        "subjectValues": [
            "uid=demo,ou=People,dc=example,dc=com"
        ]
    },
    "resourceTypeUuid": "76656a38-5f8e-401b-83aa-4ccb74ce88d2"
}" \
http://host1.example.com:8080/openam/json/realms/root/policies?_action=create
```
Response shows updated result policy json.

## OpenAM session

The `iPlanetDirectoryPro` cookie is the AM session cookie (also referred to as the session ID or SSOTokenID).

When a user successfully authenticates against an OpenAM server, a session is generated on that server.  The session contains information about the interaction between the client and the server. A decoded snippet of token is shown below
```conf
sessionID:  AQIC5wM…
maxSessionTime:  120
maxIdleTime:  30
timeLeft:  6500
userID:  bnelson
authLevel: 1
loginURL:/auth/UI/Login
service: ldapService
locale: en_US
```

Sessions are identified using a unique token called SSOTokenID. 

* The SSOToken is a C66Encoded string that identifies the session on server.  
* The Session Key is a Base64 Encoded string that identifies the location of the site, session type (such as CTS, JWT or IN_MEMORY), and the storage key used during session failover
* The period (.) is a separator.

![ssotokenid](imgs/ssotokenid.jpg "ssotokenid")

### Get session's user info

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Accept-API-Version: resource=2" \
  -H "iPlanetDirectoryPro: AQIC5wM2LY4Sfcxs...EwNDU2NjE0*" \
  "http://host1.example.com:8080/openam/json/realms/root/sessions?_action=getSessionProperties&tokenId=BXCCq...NX*1*"
```