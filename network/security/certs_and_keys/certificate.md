# Certificate

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

Creation vs Renewal:

* Creation:
generate a new cert/key which involves generating a new key (And provide your CSR to your CA).
* Renewal: 
renew you certificat which involves keeping your private key (And provide your CSR to your CA).


**Chian of trust**

Certificate Authorities (CAs) is a third-party that has already been vouched for trust by client and server. There are root CAs and intermediate CAs (any certificate that are in between CA and clients), and leaf certificate for end client.

Client and server communicate through signed leaf certificate ("signed" means trusted by intermediate/root CA).

* In the case of browser/client, builtin Object Tokens are root certificates in the default Network Security Services (NSS) database as installed on the user's PC when the user installed the software (e.g., Firefox) that uses them.

* In the case of server, such as tomcat, CA certificate should be added:
```bash
keytool -import -alias tomcat -keystore example.jks -file example.crt
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
open tomcat/conf/server.xml, and add your example.jks
```xml
<Connector port=”443″ protocol=”HTTP/1.1″
  SSLEnabled=”true”
  scheme=”https” secure=”true” clientAuth=”false”
  sslProtocol=”TLS” keystoreFile=”/your_path/yourkeystore.jks”
  keystorePass=”password_for_your_key_store” />
```

**Certificate signing request (CSR)**

A CSR is an encoded message submitted by an applicant to a CA to get an SSL certificate. CSR identifies a client by its distinguished name (DN).

A CSR is sent to a CA and CA signs this CSR and return a certificate (containing a client's public key and client DN).

Checking a CSR by
`keytool -printcertreq -file clientcsr.csr -storepass changeit`, that reveals info such as DN and key identifier.

**Signature**

A Certificate Signature (or Certificate Fingerprint) field is computed from Hash from the Cryptographic Hash Function of the whole Certificate using the identified Certificate Signature Algorithm. 

**fingerprint**

In OpenSSL the "-fingerprint" option takes the hash of the DER encoded certificate.

To check fingerprint, first convert into .der then hash it and return the result.
`openssl x509 -in cert.crt -outform DER -out cert.cer`
`sha1sum cert.cer`

**openssl/keytool examples**
* key pair generation:
`openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out private-key.pem`

* corresponding public key generation
`openssl pkey -in private-key.pem -out public-key.pem -pubout`

* generate a private key and csr
`openssl req -newkey rsa:2048 -subj "/C=US/ST=Oregon/L=Portland/O=Company Name/OU=Org/CN=www.example.com" -keyout PRIVATEKEY.key -out MYCSR.csr`

* check cert content
`keytool -printcert -file certificate.pem`

* check cert chain
`openssl s_client -connect <hostname:port> -showcerts`

* change format
```bash
keytool -importkeystore -srckeystore src_keystore.jks -destkeystore dest_keystore.p12 -srcstoretype jks deststoretype pkcs12 -srcstorepass changeit -deststorepass changeit
```

## JWK and JKWS

A JSON Web Key (JWK) is a JavaScript Object Notation (JSON) data structure that represents a cryptographic key.

A JSON Web Key Set (JWKS) is a set of keys containing the public keys used to verify any JSON Web Token (JWT) issued by the authorization server and signed using the RS256 signing algorithm.

**Some Key Fields**

* "kty" (Key Type) Parameter
* "use" (Public Key Use) Parameter, values include

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "sig" (signature)


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "enc" (encryption)
 
* "key_ops" (Key Operations) Parameter
* "alg" (Algorithm) Parameter
* "kid" (Key ID) Parameter (used to match a specific key)
* "x5u" (X.509 URL) Parameter

**A private key example**
```json
{
  "kty":"RSA",
  "kid":"juliet@capulet.lit",
  "use":"enc",
  "n":"t6Q8PWSi1dkJj9hTP8hNYFlvadM7DflW9mWepOJhJ66w7nyoK1gPNqFMSQRyO125Gp-TEkodhWr0iujjHVx7BcV0llS4w5ACGgPrcAd6ZcSR0-Iqom-QFcNP8Sjg086MwoqQU_LYywlAGZ21WSdS_PERyGFiNnj3QQlO8Yns5jCtLCRwLHL0Pb1fEv45AuRIuUfVcPySBWYnDyGxvjYGDSM-AqWS9zIQ2ZilgT-GqUmipg0XOC0Cc20rgLe2ymLHjpHciCKVAbY5-L32-lSeZO-Os6U15_aXrk9Gw8cPUaX1_I8sLGuSiVdt3C_Fn2PZ3Z8i744FPFGGcG1qs2Wz-Q",
  "e":"AQAB",
  "d":"GRtbIQmhOZtyszfgKdg4u_N-R_mZGU_9k7JQ_jn1DnfTuMdSNprTeaSTyWfSNkuaAwnOEbIQVy1IQbWVV25NY3ybc_IhUJtfri7bAXYEReWaCl3hdlPKXy9UvqPYGR0kIXTQRqns-dVJ7jahlI7LyckrpTmrM8dWBo4_PMaenNnPiQgO0xnuToxutRZJfJvG4Ox4ka3GORQd9CsCZ2vsUDmsXOfUENOyMqADC6p1M3h33tsurY15k9qMSpG9OX_IJAXmxzAh_tWiZOwk2K4yxH9tS3Lq1yX8C1EWmeRDkK2ahecG85-oLKQt5VEpWHKmjOi_gJSdSgqcN96X52esAQ",
  "p":"2rnSOV4hKSN8sS4CgcQHFbs08XboFDqKum3sc4h3GRxrTmQdl1ZK9uw-PIHfQP0FkxXVrx-WE-ZEbrqivH_2iCLUS7wAl6XvARt1KkIaUxPPSYB9yk31s0Q8UK96E3_OrADAYtAJs-M3JxCLfNgqh56HDnETTQhH3rCT5T3yJws",
  "q":"1u_RiFDP7LBYh3N4GXLT9OpSKYP0uQZyiaZwBtOCBNJgQxaj10RWjsZu0c6Iedis4S7B_coSKB0Kj9PaPaBzg-IySRvvcQuPamQu66riMhjVtG6TlV8CLCYKrYl52ziqK0E_ym2QnkwsUX7eYTB7LbAHRK9GqocDE5B0f808I4s",
  "dp":"KkMTWqBUefVwZ2_Dbj1pPQqyHSHjj90L5x_MOzqYAJMcLMZtbUtwKqvVDq3tbEo3ZIcohbDtt6SbfmWzggabpQxNxuBpoOOf_a_HgMXK_lhqigI4y_kqS1wY52IwjUn5rgRrJ-yYo1h41KR-vz2pYhEAeYrhttWtxVqLCRViD6c",
  "dq":"AvfS0-gRxvn0bwJoMSnFxYcK1WnuEjQFluMGfwGitQBWtfZ1Er7t1xDkbN9GQTB9yqpDoYaN06H7CFtrkxhJIBQaj6nkF5KKS3TQtQ5qCzkOkmxIe3KRbBymXxkb5qwUpX5ELD5xFc6FeiafWYY63TmmEAu_lRFCOJ3xDea-ots",
  "qi":"lSQi-w9CpyUReMErP1RsBLk7wNtOvs5EQpPqmuMvqW57NBUczScEoPwmUqqabu9V0-Py4dQ57_bapoKRu1R90bvuFnU63SHWEFglZQvJDMeAvmj4sm-Fp0oYu_neotgQ0hzbI5gry7ajdYy9-2lNx_76aBZoOUu9HCJ-UsfSOI8"
}
```
