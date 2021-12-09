# JWT

Json Web Token (JWT) is an open standard (RFC 7519) that defines a compact and self-contained way for securely transmitting information between parties as a JSON object. 

There are some mandatory fields in header and payload, and customised fields for tailored needs.

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
A Bearer Token is an opaque string, not intended to have any meaning to clients using it.


An example of decoded JWT 

```json
{
  "iss": "https://YOUR_DOMAIN/",
  "sub": "auth0|123456",
  "aud": [
    "my-api-identifier",
    "https://YOUR_DOMAIN/userinfo"
  ],
  "azp": "YOUR_CLIENT_ID",
  "exp": 1489179954,
  "iat": 1489143954,
  "scope": "openid profile email address phone read:appointments"
}
```

## Some Important Fields

* "iss" (issuer) claim identifies the principal that issued the JWT.

* "aud" (audience) claim identifies the recipients that the JWT is intended for.

* "exp" (expiration time) claim identifies the expiration time on or after which the JWT MUST NOT be accepted for processing.

* "iat" (issued at) claim identifies the time at which the JWT was issued.

* "jti" (JWT ID) claim provides a unique identifier for the JWT.