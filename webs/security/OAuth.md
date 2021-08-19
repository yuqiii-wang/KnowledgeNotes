# OAuth
OAuth is an open standard for authorization, commonly used as a way for Internet users to log in to third party websites using their Microsoft, Google, Facebook, Twitter, One Network etc. accounts without exposing their password.

## JWT

Json Web Token (JWT) is an open standard (RFC 7519) that defines a compact and self-contained way for securely transmitting information between parties as a JSON object. 

It contains (separated by `.`):
* Header, such as type of algo, are encoded in Base64
```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```
* Payload, lots of claims (Claims are statements about an entity (typically, the user) and additional data.) They are encoded in Base64 rather than encrypted (thus anyone can read the content).
```json
{
  "sub": "1234567890",
  "name": "John Doe",
  "admin": true
}
```
* Signature, takes Base64 encoded header, payload, and a secret, then signs by the given signing algo, 
```bash
HMACSHA256(
  base64UrlEncode(header) + "." +
  base64UrlEncode(payload),
  secret)
```

A typical usage:
`Authorization` in header of an https request for protected resources as specified in claims of this jwt. Cross-Origin Resource Sharing (CORS) won't be an issue as it doesn't use cookies
```
Authorization: Bearer <token>
```
## Token Grant 

The **authorization code** (mostly used) is obtained by using an authorization server as an intermediary between the client and resource owner.

An **access token** is a string representing an authorization issued to the client.

**Refresh tokens** are issued to the client by the authorization server and are used to obtain a new access token when the current access token becomes invalid or expires, or to obtain additional access tokens with identical or narrower scope.

### Authorization Code

* Client is an app.
* User is a human being
* Companies host auth server and resource server.

![Oauth_architecture](imgs/Oauth_architecture.jpg "Oauth_architecture")

### Client Credential

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

## OpenId Connect

**OpenID Connect (OIDC)** is a thin layer that sits on top of OAuth 2.0 that adds login and profile information about the person who is logged in.

### ID Token

Id Token is used for identify a user, profile scopes include:

* name
* gender
* birthdate
* avatar
* email

The differences from OAuth are, 

* Functionally speaking,

OpenID is about authentication (ie. proving who you are), OAuth is about authorisation (ie. to grant access to functionality/data/etc).

* Technically speaking,

In the initial request, a specific scope of openid is used, and in the final exchange the Client receives both an Access Token and an ID Token. The Access Token is thus in addition having further permission for user identity information such as email and address.

![openid_init_request](imgs/openid_init_request.png "openid_init_request")

![openid_return_token](imgs/openid_return_token.png "openid_return_token")