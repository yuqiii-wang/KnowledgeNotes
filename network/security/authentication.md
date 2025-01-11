# Authentication Methods

## Basic Auth

The username and password are concatenated with a colon `:` and encoded in Base64.
For example, the string `user:pass` is encoded as `dXNlcjpwYXNz`.

In request to server, add header `Authorization: Basic dXNlcjpwYXNz`.

If the credentials are valid, the server grants access to the resource. Otherwise, it returns another 401 Unauthorized response.

## OAuth

OAuth 2.0 is a widely used authorization framework that separates user info and credential storage and resource acquisition, hence when user requests for resources, no need to expose his/her credential, e.g., username/password.

### Typical OAuth Flow

Players:

* Authorization Server: server that stores user info and credential
* Resource Server: server that contains restricted resources, e.g., user personal photos and files
* Client: third-party apps, e.g., a phone call app wants to get user's contacts from resource server
* User/Agent: user is the actual human, agent is the tool by which user interacts with client, e.g., Chrome browser

1. Authorization Request: client to authorization server

```txt
GET /authorize?client_id=12345&response_type=code&redirect_uri=https://client.com/callback&scope=photo&state=xyz
```

where `response_type=code` means requesting for auth code; `scope=photo` means what resource to access

2. User Consent: authorization server to user agent

The Authorization Server prompts the user to log in and grant permission to the client.

3. Authorization Response: authorization server to client

If the user consents, the Authorization Server redirects the user back to the client with an authorization code.

```txt
HTTP/1.1 302 Found
Location: https://client.com/callback?code=AUTH_CODE&state=xyz
```

4. Access Token Request: client to authorization server

The client exchanges the authorization code for an access token and optionally a refresh token.
The access token is the one to use for resource server for user resource acquisition.

```txt
POST /token HTTP/1.1
Host: auth-server.com
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&code=AUTH_CODE&redirect_uri=https://client.com/callback&client_id=12345&client_secret=CLIENT_SECRET
```

5. Access Token Response: authorization server to client

The Authorization Server responds with an access token and optionally a refresh token.

```json
{
  "access_token": "ACCESS_TOKEN",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "REFRESH_TOKEN",
  "scope": "read"
}
```

6. Accessing Protected Resources: client to resource server

The client includes the access token in the Authorization header to access protected resources on the Resource Server.

```txt
GET /protected-resource HTTP/1.1
Host: resource-server.com
Authorization: Bearer ACCESS_TOKEN
```

7. Refreshing the Access Token: client to authorization server

When the access token expires, the client can use the refresh token to obtain a new access token.

```txt
POST /token HTTP/1.1
Host: auth-server.com
Content-Type: application/x-www-form-urlencoded

grant_type=refresh_token&refresh_token=REFRESH_TOKEN&client_id=12345&client_secret=CLIENT_SECRET
```

### Bearer Token

A Bearer Token is a type of access token used in authentication, typically in OAuth 2.0 and JWT (JSON Web Token)-based systems.
It is a cryptic string that represents the authorization granted to the client.

* The token is issued by an authentication server after the user successfully authenticates (e.g., by providing a username and password).
* The client includes the token in the Authorization header of HTTP requests to access protected resources.
* The token is called "bearer" because whoever "bears" (possesses) the token can use it to access the associated resources.

#### Token Validation by the Resource Server

When the Resource Server receives a Bearer Token, it:

##### Extracts the Token

Parses the token from the Authorization header (e.g., `Bearer <token>`).

##### Verifies the Signature

Uses the same secret key (for HMAC) or the public key (for RSA) to verify the token's signature.

##### Validates the Token Claims

* Checks the token's expiration time (`exp` claim).
* Verifies the issuer (`iss` claim) matches the Authorization Server.
* Ensures the audience (`aud` claim) includes the Resource Server.
* Validates other claims as needed (e.g., `sub`, `scope`).

### OpenID Connect (OIDC)

OpenID Connect (OIDC) is an identity layer built on top of OAuth 2.0.

ID Token in JWT format:

```json
{
  "iss": "https://auth-server.com",  // Issuer (Authorization Server)
  "sub": "12345",                   // Subject (user ID)
  "aud": "client-id",               // Audience (client ID)
  "exp": 1516242622,                // Expiration time
  "iat": 1516239022,                // Issued at
  "name": "John Doe",               // User's full name
  "email": "john.doe@example.com",  // User's email
  "email_verified": true            // Whether the email is verified
}
```

In the below example, having sent request to `/token` on authorization server, the response in addition contains `id_token`

```py
# Exchange the authorization code for tokens
response = requests.post(
    "https://auth-server.com/token",
    data={
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": client_secret,
    },
)

# Parse the token response
token_data = response.json()
access_token = token_data.get("access_token")
id_token = token_data.get("id_token") # id token
```

### JWT and JWKS

#### Json Web Token (JWT)

Json Web Token (JWT) is an open standard (RFC 7519) that defines a compact and self-contained way for securely transmitting information between parties as a JSON object.

It contains three parts Header, Payload and Signature:

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

Together, they are put as one JWT encoded by base64

```txt
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NSIsIm5hbWUiOiJKb2huIERvZSIsImlhdCI6MTUxNjIzOTAyMiwiZXhwIjoxNTE2MjQyNjIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
```

Decoded content:

```txt
{"alg":"HS256","typ":"JWT"}{"sub":"12345","name":"John Doe","iat":1516239022,"exp":1516242622}pDx
Lx"UP9
```

"SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c" is signature, hence the decoded byte content is not presentable as text.

##### Typical use case

`Authorization` in header of an https request for protected resources as specified in claims of this jwt.
Cross-Origin Resource Sharing (CORS) won't be an issue as it doesn't use cookies

```txt
Authorization: Bearer <token>
```

##### A More Complex Example for OAuth

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

where

* "iss" (issuer) claim identifies the principal that issued the JWT.
* "aud" (audience) claim identifies the recipients that the JWT is intended for.
* "exp" (expiration time) claim identifies the expiration time on or after which the JWT MUST NOT be accepted for processing.
* "iat" (issued at) claim identifies the time at which the JWT was issued.
* "jti" (JWT ID) claim provides a unique identifier for the JWT.
* "nonce" A random string to mitigate replay attacks.

#### JWKS (JSON Web Key Set)

A JSON Web Key Set (JWKS) is a set of cryptographic keys used to verify JWTs.

```json
{
  "keys": [
    {
      "kty": "RSA",
      "use": "sig",
      "kid": "12345",
      "alg": "RS256",
      "n": "public-key-modulus",
      "e": "public-key-exponent"
    }
  ]
}
```

##### Process

1. Key Generation

The Authorization Server generates a public/private key pair.

The public key is published in a JWKS endpoint (e.g., https://auth-server.com/.well-known/jwks.json).

2. Token Signing

The Authorization Server signs JWTs using its private key.

3. Token Verification

The Resource Server fetches the JWKS from the Authorization Server.

It uses the public key in the JWKS to verify the JWT's signature.

##### Use Example

```py
import jwt
from jwt import PyJWKClient

### JWKS endpoint
jwks_url = "https://auth-server.com/.well-known/jwks.json"
### this url returns ###
# {
#   "keys": [
#     {
#       "kty": "RSA",
#       "use": "sig",
#       "kid": "12345",
#       "alg": "RS256",
#       "n": "public-key-modulus",
#       "e": "public-key-exponent"
#     }
#   ]
# }

# Create a JWKS client
jwks_client = PyJWKClient(jwks_url)

# Example JWT
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NSIsIm5hbWUiOiJKb2huIERvZSIsImlhdCI6MTUxNjIzOTAyMiwiZXhwIjoxNTE2MjQyNjIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"

# Get the signing key from the JWKS
signing_key = jwks_client.get_signing_key_from_jwt(token)

# Decode and verify the JWT
decoded = jwt.decode(
    token,
    signing_key.key,
    algorithms=["RS256"],
    audience="client-id",
    issuer="https://auth-server.com"
)

print("Decoded JWT:", decoded)
```

## Negotiation Auth

NEGOTIATE authentication is a security mechanism used primarily in Microsoft Windows environments.

NEGOTIATE allows the client and server to negotiate the best available authentication protocols like *Kerberos* (preferred) or *NTLM* (NT LAN Manager).

### Key Features of Kerberos

* Tickets: Kerberos uses tickets (Ticket Granting Tickets - TGTs and Service Tickets) to authenticate users and services.
* Symmetric Key Cryptography: Uses shared secret keys for encryption and decryption.
* Mutual Authentication: Both the client and server verify each other's identity.

#### Ticket Granting Ticket (TGT)

A TGT is a special ticket issued by the *Key Distribution Center* (KDC) after a user successfully authenticates (e.g., by providing a username and password).

It is encrypted with the Ticket Granting Service (TGS) secret key, which only the KDC knows.

The TGT contains information about the user, such as their identity and session key, but it cannot be decrypted by the user or any other service except the KDC.

The TGT is used to request Service Tickets (STs) for accessing specific services (e.g., a file server, email server, etc.).

#### Ticket Granting Ticket (TGT) vs. OAuth Access Token

||TGT|OAuth|
|-|-|-|
|Purposes|Used in Kerberos authentication to acquire Service Tickets (STs) for accessing specific services.|Used in OAuth 2.0 to grant access to protected resources on behalf of a user or application.|
||Primarily used in enterprise environments (e.g., Active Directory).|Widely used in web-based and cloud environments (e.g., Google, Facebook, APIs).|
||The TGT is used to request Service Tickets (STs) for accessing specific services.|The access token is used directly to access protected resources.|
||Designed for enterprise single sign-on (SSO) in environments like Active Directory.|Designed for delegated authorization in web and cloud applications (e.g., APIs).|
|Issuance|Issued by the Key Distribution Center (KDC) after the user authenticates (e.g., via username/password).|Issued by the Authorization Server after the user or application grants consent.|
||The TGT is encrypted with the Ticket Granting Service (TGS) secret key.|The access token is typically a JSON Web Token (JWT) or an opaque string.|
||Relies on symmetric key cryptography (shared secrets between the KDC and services).|Relies on asymmetric cryptography (public/private keys) or bearer tokens.|

## Keepie Auth

Keepie is a "device"-based authentication method that on request for restricted resource access, the Keepie authentication ALWAYS and ONLY grants access to already registered "devices".

The "device" is an abstract concept that refers to non-changeable entities, e.g., IP address, hardware device MAC address, user account.

### Typical User Process

1. When a user attempts to login, Keepie server generates a secure, temporary token or key
2. The Keepie server checks against authorization server what "devices" to bind
3. From now on, the Keepie token will always be sent to the bound devices
4. User device having received the Keepie token, requests for restricted resource access with this Keepie token
5. If the user switched to using another not-yet-registered device, the new device will not receive any Keepie token, hence not authorized to access restricted resources

### Use Example in Multi-Server Cloud

Only pre-defined servers (devices) have privileges to request access for restricted resources.

1. Set up a config/authorization server that stores a privilege map that what servers have what privileges on what resources
2. When a application server requests for resources on a resource server, this resource server sends Keepie tokens to privileged servers as per defined in the config server
3. Only if this application server is set up with the privileged access, it shall receive the Keepie token, and make requests with the received Keepie token
4. The resource server checks the sent Keepie token for authorized privileges, if permitted then shall return resources to the application server
