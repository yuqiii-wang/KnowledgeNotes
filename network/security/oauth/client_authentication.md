# Client Authentication

Methods available on ForgeRock. Only one auth method can be used as specified in client profile (e.g., Software Statement Assertion) via the config `tokenEndPointAuthMethod`.

## 1. By ` client_id` and `client_secret`

```bash
curl \
--request POST \
--data "client_id=myClient" \
--data "client_secret=forgerock" \
...
```

## 2. Using Authorization Headers

Clients that have a client secret can send the client ID and the secret in a basic authorization header with the base64-encoded value of client_id:client_secret.

```bash
curl \
--header "Authorization: Basic bXlDbGllbnQ6Zm9yZ2Vyb2Nr" \
--request POST \
...
```

## 3. Using JWT Profiles

Clients can send a signed JWT to the authorization server as credentials instead of the client ID and/or secret, as per (RFC 7523) JWT Profile for OAuth 2.0 Client Authentication and Authorization Grants. 

![oauth2-jwt-bearer-authn](imgs/oauth2-jwt-bearer-authn.svg "oauth2-jwt-bearer-authn")

* aud. Specifies the authorization server
* sub. Client ID
* iss the client, or a third party. 

**-> must configure the public key or HMAC secret** in the client profile so AM can validate it: 

* Configuring Certificates Represented as PEM Files
```bash
-----BEGIN CERTIFICATE-----
MIIDETCCAfmgAwIBAgIEU8SXLjAN...
-----END CERTIFICATE-----
```

* Configuring Public Keys in JWK Format
```json
{
  "keys": [
    {
      "alg": "RSA-OAEP-256",
      "kty": "RSA",
      "use": "sig",
      "kid": "RemA6Gw0...LzsJ5zG3E=",
      "n": "AL4kjz74rDo3VQ3Wx...nhch4qJRGt2QnCF7M0",
      "e": "AQAB"
    }
  ]
}
```

## 4. Using Mutual TLS

The certificate presented by the client contains a subject distinguished name that matches exactly a value specified in the client profile in AM.

1. Import clients' CA and CA intermediate certs into AM trsuted secret store.
2. Mapping imported certs to `am.services.oauth2.tls.client.cert.authentication` secret ID on AM server
3. Add the subject distinguished name that must appear in the client certificate to be able to authenticate