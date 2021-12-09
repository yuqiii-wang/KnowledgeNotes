# User Consent

When a client requests a scope combination, AM checks if the user has already consented each scope within the combination. If AM can find the scopes across multiple saved consent entries, AM will not require the user to consent. If part of the requested scope combination is not found in any entry, AM will require the user to consent. 


## Remote Consent Service

Remote Consent Service allow the consent-gathering part of an OAuth 2.0 flow to be handed off to a separate service. 

During an OAuth 2.0 flow that requires user consent, AM can create a consent request JWT that contains the necessary information to render a consent gathering page. RCS decrypts the JWT, and renders the consent page to the resource owner.

Example Request JWT 
```json
{
  "clientId": "myClient",
  "iss": "https://openam.example.com:8443/openam/oauth2",
  "csrf": "gjeH2C43nFJwW+Ir1zL3hl8kux9oatSZRso7aCzI0vk=",
  "client_description": "",
  "aud": "rcs",
  "save_consent_enabled": false,
  "claims": {},
  "scopes": {
      "write": null
  },
  "exp": 1536229486,
  "iat": 1536229306,
  "client_name": "My Client",
  "consentApprovalRedirectUri": "https://openam.example.com:8443/openam/oauth2/authorize?client_id=MyClient&response_type=code&redirect_uri=https://application.example.com:8443/callback&scope=write&state=1234zy",
  "username": "demo"
}
```

Explained:
* `iss`: OAuth 2.0 Provider Service in AM (Auth Server)
* `aud`: remote consent service (RCS): (Client APP) renders a consent page, gathers the result, signs and encrypts the result, and returns it to the authorization server (e.g., could be middleware aws lambda that renders an html page)
* `csfr`: a unique string that must be returned in the response to help prevent cross-site request forgery (CSRF) attacks; AM generates this string from a hash of the user's session ID.
* `claims`: additional info such as prompt text for user to help decide if consent should be given.

After User taking action, a response is generated containing a JWT, returned to AM server.

Example Response JWT 
```json
{
  "clientId": "myClient",
  "iss": "rcs",
  "csrf": "gjeH2C43nFJwW+Ir1zL3hl8kux9oatSZRso7aCzI0vk=",
  "client_description": "",
  "aud": "https://openam.example.com:8443/openam/oauth2",
  "save_consent_enabled": false,
  "claims": {},
  "scopes": "[write]",
  "exp": 1536229430,
  "iat": 1536229250,
  "client_name": "My Client",
  "consentApprovalRedirectUri": "https://openam.example.com:8443/openam/oauth2/authorize?client_id=MyClient&response_type=code&redirect_uri=https://application.example.com:8443/callback&scope=write&state=1234zy",
  "username": "demo",
  "decision": true
}
```