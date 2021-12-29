
# A Walk-through example (by openssl/keytool)

* CA side:

A CA needs a key pair for signing client CSR.
```bash
openssl req -x509 -sha256 -nodes -days 365 -newkey rsa:2048 -keyout ca-key.pem -out ca-cert.pem -subj "/C=CN/ST=Shenzhen/L=Shenzhen/O=exampleca/CN=exampleca" 
```

* Client side:

get a key pair
```bash
keytool -genkey -keyalg RSA -alias exampleclient -keystore keystore.jks -storetype jks -storepass changeit -keypass changeit -validity 365 -keysize 2048 -dname 'CN=www.exampleclient.com,OU=examplecompany,O=examplecompany,ST=Shenzhen,C=CN'
```

check the key pair
```bash
keytool -v -list -keystore keystore.jks -storepass changeit
```

get a csr
```bash
keytool -certreq -alias exampleclient -keystore keystore.jks -keyalg RSA -storepass changeit -file exampleclientcsr.csr
```

check the csr
```bash
keytool -printcertreq -storepass changeit -file exampleclientcsr.csr
```

* CA side:

signing csr
```bash
openssl x509 -req -CA ca-cert.pem -CAkey ca-key.pem -in exampleclientcsr.csr -out exampleclient.cer -days 365 -CAcreateserial
```

* Client side:

import both ca root/intermediate and client signed certs to client's keystore
```bash
keytool -import -keystore keystore.jks -file ca-cert.pem -alias ca -storepass changeit

keytool -import -keystore keystore.jks -file exampleclient.cer -alias exampleclient-signed -storepass changeit
```
### ====================================================
### if not requiring cert been signed, you can just export the cert from client's keystore
```bash
keytool -export -keystore keystore.jks -file exampleclient-selfsigned.cer -alias exampleclient -storepass changeit -rfc
```
