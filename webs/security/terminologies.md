# Some Terminologies

## CVE

**CVE**, short for Common Vulnerabilities and Exposures, is a list of publicly disclosed computer security flaws. When someone refers to a CVE, they mean a security flaw that's been assigned a CVE ID number.

## Hardening

**Hardening** is a process that prevents possible cyber attacks on servers, such as AMI.

Security group rules: A security rule applies either to inbound traffic (**ingress**) or outbound traffic (**egress**). 

## TCP Handshake

1. client sends SYN=1 with SEQ=x to server
2. server replies with ACK=1 and PACK=x+1 to indicate connection signal reception, and sends its SYN=1 and SEQ=y to the client
3. client replies to server with ACK=1 and PACK=y+1 that it acknowledges the server's attempt to connect

x and y here indicate connection acknowledge sync.

![tcp-three-way-handshake](imgs/tcp-three-way-handshake.gif "tcp-three-way-handshake")

## SSL 

`SSH tunneling` is a method of transporting arbitrary networking data over an encrypted SSH connection.

**SSL** stands for Secure Socket Layer.

The client contacts the server and sends the first message. The first message causes the client and server to exchange a few messages to negotiate the encryption algorithm to use and to pick an encryption key to use for this connection. Then the client's data is sent to the server. After this is done the client and the server can exchange information at will.

An asymmetric cryptography:

![asymmetric_cryptography](imgs/asymmetric_cryptography.png "asymmetric_cryptography")

A symmetric cryptography:

![symmetric-cryptography](imgs/symmetric-cryptography.png "symmetric-cryptography")

An SSL handshake:

1. Hello messages contain SSL version number, cipher settings, session-specific data, etc. (browser receives the certificate served as a **public key**)
2. The client verifies the server's SSL certificate from **CA (Certificate Authority)** and authenticates the server (CA uses **private key** to perform verification)
3. The client creates a session key, encrypts it with the server's public key and sends it to the server
4. The server decrypts the session key with its private key and sends the acknowledgement

![ssl-handshack](imgs/ssl-handshack.png "ssl-handshack")

After a successful handshake, data is transferred with session key encryption.



##OAuth
OAuth is an open standard for authorization, commonly used as a way for Internet users to log in to third party websites using their Microsoft, Google, Facebook, Twitter, One Network etc. accounts without exposing their password.

![Oauth_architecture](imgs/Oauth_architecture.jpg "Oauth_architecture")

**jwt**

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

## LDAP

**LDAP** (lightweight directory access protocol) is a communication protocol that defines the methods in which a directory service can be accessed.

Entries are all added to an LDAP system as branches on trees called **Data Information Trees**, or DITs.

```
dn: cn=John Doe, ou=people, dc=example,dc=com
objectclass: person
cn: John Doe
givenName: Doe
sn: Doe
telephoneNumber: +1 888 555 6789
telephoneNumber: +1 888 555 1232
mail: john@example.com
manager: cn=Barbara Doe,dc=example,dc=com
objectClass: inetOrgPerson
objectClass: organizationalPerson
objectClass: person
objectClass: top
```

Given the example below, (`dn` for Distinguished Name) `sn=Doe` is an entry. The direct parent is an entry called `ou=people` which is probably being used as a container for entries describing people. The parents of this entry derived from the `example.com` domain name, which functions as the root of our DIT.

`ObjectClass` attribute specifies the object classes of an entry, which (among other things) are used in conjunction with the controlling schema to determine the permitted attributes of an entry.

Here below is an example of `ObjectClass` definition.
```
( 2.5.6.6 NAME 'people' SUP top STRUCTURAL MUST ( sn $ cn $ ou) MAY ( userPassword $ telephoneNumber $ seeAlso $ description ) )
```
This indicates that the people object class is a structural class with `OID 2.5.6.6`. Entries with the people object class are required to include the `sn` (surname), `ou` (organizational unit) and `cn` (common name) attribute types, and may also include any or all of the userPassword, telephoneNumber, seeAlso, and description attribute types. And because the people object class inherits from the `top` object class, entries containing the people object class are required to also include the objectClass attribute type (which is declared as mandatory in the top object class). The people object class is not obsolete and does not have a description.

## TLS

Transport Layer Security (TLS) is a successor of SSL, with most publically visible use case of https.

TLS is different from SSL in terms of 
1. Cipher suites
2. Alert messages
3. Hash algos
4. Certificate format 

**CRL** 

a certificate revocation list (or CRL) is "a list of digital certificates that have been revoked by the issuing certificate authority (CA) before their scheduled expiration date and should no longer be trusted".

**TLS Certificate**

X.509 is a standard defining the format of public key certificates.

## Certificate

A certificate is a container of a public key, with added info such as issuer, experation time, encryption algo, etc:

* Version: A value (1, 2, or 3) that identifies the version number of the certificate
* Serial Number: A unique number for each certificate issued by a CA
* CA Signature Algorithm: Name of the algorithm the CA uses to sign the certificate contents
* Issuer Name: The distinguished name (DN) of the certificate's issuing CA
* Validity Period: The time period for which the certificate is considered valid
* Subject Name: Name of the entity represented by the certificate
* Subject Public Key Info: Public key owned by the certificate subject

Some common used extensions include:

* .crt, .pem - (Privacy-enhanced Electronic Mail) Base64 encoded DER certificate, enclosed between "-----BEGIN CERTIFICATE-----" and "-----END CERTIFICATE-----"
* .der, .cer - usually in binary DER  (Distinguished Encoding Rules) form

**Chian of trust**

Certificate Authorities (CAs) is a third-party that has already been vouched for trust by client and server. There are root CAs and intermediate CAs (any certificate that are in between CA and clients), and leaf certificate for end client.

**Certificate signing request (CSR)**

A CSR is an encoded message submitted by an applicant to a CA to get an SSL certificate. CSR identifies a client by its distinguished name (DN).

A CSR is sent to a CA and CA signs this CSR and return a certificate (containing a client's public key and client DN) and a client private key.

**Signature**

A Certificate Signature (or Certificate Fingerprint) field is computed from Hash from the Cryptographic Hash Function of the whole Certificate using the identified Certificate Signature Algorithm. 

**fingerprint**

In OpenSSL the "-fingerprint" option takes the hash of the DER encoded certificate.

To check fingerprint, first convert into .der then hash it and return the result.
`openssl x509 -in cert.crt -outform DER -out cert.cer`
`sha1sum cert.cer`

**openssl examples**
* private key generation:
`openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out private-key.pem`

* corresponding public key generation
`openssl pkey -in private-key.pem -out public-key.pem -pubout`

* generating CSR
`openssl req -newkey rsa:2048 -subj "/C=US/ST=Oregon/L=Portland/O=Company Name/OU=Org/CN=www.example.com" -keyout PRIVATEKEY.key -out MYCSR.csr`

* check cert chain
`openssl s_client -connect <hostname:port> -showcerts`