# Opendj CTS

The ForgeRock Core Token Service (CTS) is a specific DS repository used to store and persist AM SSO (Aceess Management Single Sign On), OAuth2/OIDC and SAML (Security Assertion Markup Language) tokens.



## Example Checks

Filter on `coreTokenString03=<user>` and `coreTokenString10=refresh_token` to find OAuth2 refresh tokens for a `<user>`
```bash
./ldapsearch --hostname ds1.example.com --port 1389 --bindDN uid=admin --bindPassword password --baseDN "ou=famrecords,ou=openam-session,ou=tokens,dc=openam,dc=forgerock,dc=org" "(&(coreTokenString03=demo)(coreTokenString10=refresh_token))"
```

## CTS Stored Token Examples

```bash
dn: coreTokenId=4e915f7a-08ec-4c65-915f-2256d6c3a503,ou=famrecords,ou=openam-session,ou=tokens,dc=openam,dc=forgerock,dc=org
objectClass: top
objectClass: frCoreToken
coreTokenObject: {"redirectURI":["http://example.com"],"clientID":["OIDCclient1"],"ssoTokenId":["mJLebOGs9Y4rAE_JY0uSaS_SVwM.*AAJTSQACMDEAAlNLABwvbWJRSVJ4aGdVcUhHTmNUTkRZVjAxcVl4eFE9AAJTMQAA*"],"auditTrackingId":["a7180708-c39b-4f92-90ea-b2b8bb79ec75-83912"],"tokenName":["access_code"],"authModules":["DataStore"],"code_challenge_method":[],"userName":["demo"],"nonce":["abcdef"],"authGrantId":["f58f19f9-7f3f-43db-be90-466643414143"],"acr":[],"expireTime":["1523281431770"],"scope":["openid","profile"],"claims":[null],"realm":["/myRealm"],"id":["4e915f7a-08ec-4c65-915f-2256d6c3a503"],"state":[],"tokenType":["Bearer"],"code_challenge":[],"issued":["true"]}
coreTokenString11: abcdef
coreTokenString01: openid,profile
coreTokenString10: access_code
coreTokenString04: http://example.com
coreTokenString15: f58f19f9-7f3f-43db-be90-466643414143
coreTokenString03: demo
coreTokenExpirationDate: 20180409134351.770Z
coreTokenString08: /myRealm
coreTokenString09: OIDCclient1
coreTokenId: 4e915f7a-08ec-4c65-915f-2256d6c3a503
coreTokenString06: true
coreTokenString07: Bearer
coreTokenType: OAUTH

```