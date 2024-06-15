# Access Management

Access management is about controlling access to resources using two processes: *authentication* and *authorization*.

|Party of Interest|Explain|
|-|-|
|Client|An application|
|Browser/User agent|End user|
|Resource Server (RS)|A server that holds protected resources, such as user profiles|
|Access Management (AM) server|A server that defines access rules as well as manages tokens|

For example, WeChat has an RS that stores users' info, e.g., name, birthday, gender, and has an AM that authorizes various registered client apps to use WeChat user info, e.g., a game app that wants to load user info rather than asking users to re-register his/her name.

## Authentication

Authentication determines who is trying to access a resource. For example, AM issues an SSO token to a user/browser that helps identify permitted resources.

When a user successfully authenticates, AM creates a session, which allows AM to manage the user's access to resources. The session is assigned an *authentication level*, by which resources of different security levels are granted access.

AM supports delegated authentication through third-party identity providers, typically:

|Delegates|
|-|
|OpenID Connect 1.0|
|OAuth 2.0|
|Facebook|
|Google|
|WeChat|

### Multi-Factor Authentication

An authentication technique that requires users to provide multiple forms of identification when logging in to AM.

### Sessions

AM sets session cookies either in user's browser/end device (client based) or inside server (server based).

A session ends typically for these scenarios:

* When a user explicitly logs out
* When an administrator monitoring sessions explicitly terminates a session
* When a session exceeds the maximum time-to-live or being idle for too long

AM should invalidate cookies when terminating a session.

### Single Sign-On

Single sign-on (SSO) lets users who have authenticated to AM access multiple independent services from a single login session by storing user sessions as HTTP cookies.

Cross-domain single sign-on (CDSSO) provides SSO inside the same organization within a single domain or across domains.

Web Agents and Java Agents wrap the SSO session token inside an OpenID Connect (OIDC) JSON Web Token (JWT). 

In general, this flow works as

| |CDSSO Flow|
|-|---|
|1.|Browser requests access to a protected resource on RS|
|2.|RS either sets an SSO cookie then redirects to AM server, or directly redirects to AM server (to `authorize` endpoint)|
|3.|AM server either sets an SSO token and validates it, or just validates the token|
|4.|If AM found token not valid, it asks for authentication then goes back to the 3rd step authorizing the expecting SSO token; The SSO token must be valid to proceed|
|5.|AM server `authorize` endpoint responds with an OIDC-embedded SSO token to browser|
|6.|Browser presents this SSO token to RS, who relays this token to AM server for validation check|
|7.|AM server responds to RS with either allowed or denied, and RS then responds to browser either with requested resources or a rejection message|

## Authorization

Authorization determines whether to grant or deny access to requested resource by defined rules.

### Resource Types

* URL resource type:

What URLs (matched by RE such as `*://*:*/*?*`) are permitted access by what action (such as `GET`, `POST`, `PUT`).

* OAuth2 Scope resource type

What scopes are permitted. These usually are AM admin defined with semantic significance. For example, in Open Banking, typical scopes are *account:read*, *account:update*, *account:balance:read*.

### Authorization Header

Syntax: `Authorization: <auth-scheme> <authorization-parameters>`

* Basic

```txt
Authorization: Basic dXNlcm5hbWU6cGFzc3dvcmQ=
```

where `dXNlcm5hbWU6cGFzc3dvcmQ=` is the Base64 encoded string of `username:password`.

* Bearer Token

```txt
Authorization: Bearer 2YotnFZFEjr1zCsicMWpAA
```

where `2YotnFZFEjr1zCsicMWpAA` is an access token granted by an AM server (commonly used with OAuth 2.0).

* Digest

```txt
Authorization: Digest username="Mufasa",
                realm="testrealm@host.com",
                nonce="dcd98b7102dd2f0e8b11d0f600bfb0c093",
                uri="/dir/index.html",
                response="e524d89ebc8e6480f296f0bb20b2232b",
                opaque="5ccc069c403ebaf9f0171e9517f40e41"
```

Digest is a more secure method of authentication compared to Basic, used cryptographic hashing.

### Policy Sets

Policy sets define implementation of rules, checking if a request on behalf of an end user, has privileges accessing a particular resource.

Some most typical are, checking if a user can only do `GET` not `POST`, resource url pattern matches his privileged access resource list, etc.
