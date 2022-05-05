# Cert generation example

## A Walk-through example (by openssl/keytool)

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
### if not requiring cert been signed, you can just export the cert from client's keystore
```bash
keytool -export -keystore keystore.jks -file exampleclient-selfsigned.cer -alias exampleclient -storepass changeit -rfc
```


## A Walk-through example (by conf file)

1. Create a `server-csr.conf`, in which the server dn is defined.
```conf
[ req ]
default_bits = 2048
encrypt_key = no
default_md = sha256
utf8 = yes
string_mask = utf8only
prompt = no
distinguished_name = server_dn
req_extensions = server_reqext
[ server_dn ]
commonName = threatshield.example.com 
[ server_reqext ]
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth,clientAuth
subjectKeyIdentifier = hash
subjectAltName = @alt_names
[alt_names]
DNS.1 = threatshield.example.com
```

2. Certificate Signing Request, in which you obtain a private key `server.key` and a public key (aka a cert) `server.csr`
```bash
openssl req -new -config server-csr.conf -out server.csr \
        -keyout server.key
```

3. Create a `CA.conf`

```conf
[ ca ]
default_ca = the_ca
[ the_ca ]
dir = ./CA
private_key = $dir/private/CA.key
certificate = $dir/CA.crt
new_certs_dir = $dir/certs
serial = $dir/db/crt.srl
database = $dir/db/db
unique_subject = no
default_md = sha256
policy = any_pol
email_in_dn = no
copy_extensions = copy
[ any_pol ]
domainComponent = optional
countryName = optional
stateOrProvinceName = optional
localityName = optional
organizationName = optional
organizationalUnitName = optional
commonName = optional
emailAddress = optional
[ leaf_ext ]
keyUsage = critical,digitalSignature,keyEncipherment
basicConstraints = CA:false
extendedKeyUsage = serverAuth,clientAuth
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid:always
[ ca_ext ]
keyUsage                = critical,keyCertSign,cRLSign
basicConstraints        = critical,CA:true,pathlen:0
subjectKeyIdentifier    = hash
authorityKeyIdentifier  = keyid:always
```

4. To sign the server's certificate by ca
```bash
openssl ca -config CA.conf -days 365 -create_serial \
    -in server.csr -out server.crt -extensions leaf_ext -notext
```

5. Link certificates together to have the certificate chain in one file
```bash
cat server.crt CA/CA.pem >server.pem
```
