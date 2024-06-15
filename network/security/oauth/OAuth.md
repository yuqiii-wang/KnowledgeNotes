# OAuth

OAuth is an open standard for authorization, commonly used as a way for Internet users to log in to third party websites using their Microsoft, Google, Facebook, Twitter, One Network etc. accounts without exposing their password.

## A Typical OAuth Flow

* Client is an app.
* User is a human being
* Companies host auth server and resource server, e.g., Google

1. Client asks for authorization.

```txt
GET /authorize?response_type=code&client_id=CLIENT_ID&redirect_uri=REDIRECT_URI&scope=read_profile&state=xyz HTTP/1.1
Host: authorization-server.com
```

Access management server responds with an authorization code (usually one or two minute expiration time)

```txt
HTTP/1.1 302 Found
Location: https://client-app.com/callback?code=AUTHORIZATION_CODE&state=xyz
```

2. Client asks for access token with the obtained authorization code.

```txt
POST /token HTTP/1.1
Host: authorization-server.com
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&code=AUTHORIZATION_CODE&redirect_uri=REDIRECT_URI&client_id=CLIENT_ID&client_secret=CLIENT_SECRET
```

Access management server responds with an access token, and an optional refresh token.

```txt
HTTP/1.1 200 OK
Content-Type: application/json

{
  "access_token": "ACCESS_TOKEN",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "REFRESH_TOKEN"
}
```

3. Client takes the obtained access token to request for user info from resource server.

```txt
GET /resource HTTP/1.1
Host: resource-server.com
Authorization: Bearer ACCESS_TOKEN
```

Resource server

```txt
HTTP/1.1 200 OK
Content-Type: application/json

{
  "user_id": "12345",
  "name": "John Doe",
  "email": "john.doe@example.com"
}
```

## OpenId Connect

**OpenID Connect (OIDC)** is a thin layer that sits on top of OAuth 2.0 that adds login and profile information about the person who is logged in.

### A Typical OpenID Flow

1. Client asks for authorization

```txt
GET /authorize?response_type=code&client_id=CLIENT_ID&redirect_uri=REDIRECT_URI&scope=openid%20profile%20email&state=xyz&nonce=abc HTTP/1.1
Host: openid-provider.com
```

Access management server responds with the requested authorization code.

```txt
HTTP/1.1 302 Found
Location: https://client-app.com/callback?code=AUTHORIZATION_CODE&state=xyz
```

2. Client asks for access token with the obtained authorization code

```txt
POST /token HTTP/1.1
Host: openid-provider.com
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&code=AUTHORIZATION_CODE&redirect_uri=REDIRECT_URI&client_id=CLIENT_ID&client_secret=CLIENT_SECRET
```

Access management server responds with an access token and optionally a refresh token.

Notice here `ID_TOKEN` is provided compared to the typical OAuth access token response.

```txt
HTTP/1.1 200 OK
Content-Type: application/json

{
  "access_token": "ACCESS_TOKEN",
  "token_type": "Bearer",
  "expires_in": 3600,
  "id_token": "ID_TOKEN",
  "refresh_token": "REFRESH_TOKEN"
}
```

3. Client decodes the `ID_TOKEN`, to read and verify user info.

`ID_TOKEN` is a jwt, where fields are separated by `.`:

* Header
* Payload
* Signature

For example,

```txt
eyJhbGciOiJSUzI1NiIsImtpZCI6IjE2ZGVkZTJjZGEifQ.eyJpc3MiOiJodHRwczovL29wZW5pZC1wcm92aWRlci5jb20iLCJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiZW1haWwiOiJqb2huLmRvZUBleGFtcGxlLmNvbSIsInBpY3R1cmUiOiJodHRwczovL2V4YW1wbGUuY29tL2pvaG4uanBnIiwibm9uY2UiOiJhYmMxMjMiLCJhdWQiOiJDTElFTlRfSUQiLCJleHAiOjE2ODc2MjQyMDAsImlhdCI6MTY4NzYyMDYwMH0.DHvhxDHN8CpA3RpR1zU93X1S-sX3OyLvN3cF9bn5ZsAg99F-9G9R7I5d69Kc16CeW3kF5TBr5j6U4KwZ1lbvcJ8tfQdntX5LKsa1FsC4PZXcscIN3x1wHVg5hKkAwN5whV3-7xsjYjPEfxFJHcfFwbZndLQWXN1zPqB-d8tIlo
```

where the header `eyJhbGciOiJSUzI1NiIsImtpZCI6IjE2ZGVkZTJjZGEifQ` is base64 encoded json about signature encryption algo.

```json
{
  "alg": "RS256",
  "kid": "16dede2cda"
}
```

The payload is also a base64 encoded json about user info.

```json
{
  "iss": "https://openid-provider.com",
  "sub": "1234567890",
  "name": "John Doe",
  "email": "john.doe@example.com",
  "picture": "https://example.com/john.jpg",
  "nonce": "abc123",
  "aud": "CLIENT_ID",
  "exp": 1687624200,
  "iat": 1687620600
}
```

The signature is used to verify the payload info by corresponding public key.

## Token Grant

The **authorization code** (mostly used) is obtained by using an authorization server as an intermediary between the client and resource owner.

An **access token** is a string representing an authorization issued to the client.

**Refresh tokens** are issued to the client by the authorization server and are used to obtain a new access token when the current access token becomes invalid or expires, or to obtain additional access tokens with identical or narrower scope.

### access token vs refresh token

Section 4.2 of the draft OAuth 2.0 protocol indicates that an authorization server can return both an access_token (which is used to authenticate oneself with a resource) as well as a refresh_token, which is used purely to create a new access_token.

### Authorization Code

#### Authorization Code Grant Flow

![oauth2-authz](imgs/oauth2-authz.svg "oauth2-authz")

#### Authorization Code Grant Flow with PKCE

The flow is similar to the regular Authorization Code grant type, but the client must generate a code that will be part of the communication between the client and the authorization server.

Proof Key for Code Exchange (PKCE)

![oauth2-authz](imgs/oauth2-authz-pkce.svg "oauth2-authz")

### Client Credentials

Client Credentials grant is used when the client is also the resource owner and it is accessing its own data instead of acting in behalf of a user.
In other words, client represents user requesting and managing user info.

Client provided credentials, used as API key, etc.

For example,

```bash
curl -X POST https://localhost:8080/.../access_token
  -H "Content-Type: application/x-www-urlencoded" \
  -H "Acecept: 1.0" \
  --data-urlencoded "grant_type=client_credentials" \
  --data-urlencoded "client_id=myid" \
  --data-urlencoded "client_secret=abc123" \
  --data-urlencoded "scope=basic email"
```

If succeeded, auth server should reply an access token.

```json
HTTP/1.1 200 OK
Content-Type: application/json;charset=UTF-8
Cache-Control: no-store
Pragma: no-cache

{
 "access_token":"2YotnFZFEjr1zCsicMWpAA",
 "token_type":"example",
 "expires_in":3600,
 "example_parameter":"example_value"
}
```

### Use of Nonce

A nonce is an arbitrary number that can be used just only once in a cryptographic.

This used to deal with access_token transmission (stateless authentication) between client and server, specially when request contains cookies.

* The nonce is generated after authentication whether first API after authentication or separate API

* Nonce JWT is generated from username, clientID which should be provided by client itself and the Not before claim set.

![nonce_process_flow](imgs/nonce_process_flow.png "nonce_process_flow")

### Use of State

Reference:

https://auth0.com/docs/configure/attack-protection/state-parameters

As described in rfc6749, state parameter preserves some state objects set by the client in the Authorization request and makes it available to the client in the response to mitigate CSRF attacks.

Some typical generation can be

```py
state = hash('sha256', session_id())
```

```py
state = hash('sha256', client_id())
```

Init by a client for authorization to an AM server.

```bash
curl \
'https://oauth.pipedrive.com/oauth/authorize?client_id=xxx&redirect_uri=https://xxx.com/oauth2/redirect.html&state=xyzABC123'
```

This client checks if response from the AM server has same `state=xyzABC123`.
