# Open ID Connect

OIDC (Open ID Connect) is a specification on top of OAuth with additional request with `scope=openid` and response from Auth Server with `id_token=$id_token_jwt` which illustrates a user identity info (such as name, email, etc.).

## OIDC Authorization Code Grant Flow Example

1. Resource Owner access to Client APP and redirect to Auth Server

From client's app
```bash
curl https://example.auth.server.com/oauth2/authorize?
    nonce=$nonce&
    state=$state&
    scope:openid%20profile&
    client_id=$client_id&
    response_type=code&
    redirect_url=https://example.client.app.com/redirect_1.html
```

2. Auth Server permits authorization (Remote Consent Service) and responds to client

Response (a url link)
```bash
https://example.auth.server.com/oauth2/authorize?
 scope=openid&
 response_type=code&
 code=4/P7q7W91a-oMsCeLvIaQm6bTrgtp7&
 nonce=$nonce&
 state=$state&
 redirect_uri=https://example.client.app.com/redirect_1.html
 client_id=$client_id
```

3. Client APP's request for access token

```bash
curl https://example.auth.server.com/oauth2/access_token
code=4/P7q7W91a-oMsCeLvIaQm6bTrgtp7&
client_id=$client_id&
client_secret=$client_secret&
redirect_uri=ttps://example.client.app.com/redirect_1.html&
grant_type=authorization_code
```

4. Response from Auth server with access token
```json
{
  "access_token": "1/fFAGRNJru1FTz70BzhT3Zg",
  "expires_in": 3920,
  "token_type": "Bearer",
  "scope": "openid%20profile",
  "refresh_token": "1//xEoDL4iW3cxlI7yDbSRFYNG01kVKM2C-259HOF2aQbI",
  "id_token": "id_token_jwt"
}
```

Decoded `$id_token_jwt`
```json
{
  "iss": "https://example.auth.server.com",
  "aud": "https://example.client.app.com",
  "sub": "10769150350006150715113082367",
  "at_hash": "HK6E_P6Dh8Y93mRNtsDB1Q",
  "email": "jsmith@example.com",
  "name": "Jason Smith",
  "email_verified": "true",
  "iat": 1353601026,
  "exp": 1353604926,
  "nonce": "0394852-3190485-2490358"
}
```

Explained:
* aud: OAuth 2.0 client ID
* iss: Issuer Identifier for the Issuer of the response
* sub: An identifier for the user