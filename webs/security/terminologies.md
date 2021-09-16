# Some Terminologies

## CVE

**CVE**, short for Common Vulnerabilities and Exposures, is a list of publicly disclosed computer security flaws. When someone refers to a CVE, they mean a security flaw that's been assigned a CVE ID number.

## Hardening

**Hardening** is a process that prevents possible cyber attacks on servers, such as AMI.

Security group rules: A security rule applies either to inbound traffic (**ingress**) or outbound traffic (**egress**). 

## Log in vs Log on

Log in is a verb phrase. When you log in to something, you provide credentials to access material.

Log on is that, when you log on to something, you are simply accessing digital material, without necessarily needing to provide credentials.

## Basic vs Bearer

* Basic

In the context of an HTTP transaction, **basic access authentication** (specified in RFC 7617) is a method for an HTTP user agent (e.g. a web browser) to provide a user name and password when making a request. In basic HTTP authentication, a request contains a header field in the form of `Authorization: Basic <credentials>`, where credentials is the Base64 encoding of ID and password joined by a single colon `:`.

A client sends a request with a header.
```bash
Authorization: Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==
```
BASE64 decoded: *Aladdin:open sesame*

Given wrong username/password, `401 Unauthorized` is returned.

* Bearer

Bearer token is a type of encryption token `Authorization: Bearer <token>` defined in RFC6750 commonly used in OAuth.

Example usages are:
1. authorization code
2. access token
3. refresh token

Authorization code is granted via *redirect_uri* to avoid hacker implicitly redirecting end user to hacker's endpoint (Cross-site request forgery (CSRF)), such as
```bash
 HTTP/1.1 302 Found
Location: http://example.com:8080/oauth?code=7rQNrZ2CpyA8dshIJFn8SX43dAk&scope=openid%20profile&iss=http%3A%2F%2Fhost1.example.com%3A8080%2Fopenam%2Foauth2&state=af0ifjsldkj&client_id=myClientID
```