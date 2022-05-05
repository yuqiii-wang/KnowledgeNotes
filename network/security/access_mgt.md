# Access Management

Access management is about controlling access to resources using two processes: *authentication* and *authorization*.

|Party of Interest|Explain|
|-|-|
|Browser/User agent|End user|
|Resource Server (RS)|A server that holds protected resources, such as user profiles|
|Access Management (AM) server|A server that defines access rules as well as manages tokens|

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

### Policy Sets

Policy sets define implementation of rules, checking if a request on behalf of an end user, has privileges accessing a particular resource.

Some most typical are, checking if a user can only do `GET` not `POST`, resource url pattern matches his privileged access resource list, etc.