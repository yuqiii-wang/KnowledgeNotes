# Postman

## Common Setup Debugging 

### Certs

* **Host** Be aware of empty space 

* **CRT** Your PEM certificate. The certificate must be a Base64-encoded ASCII file and contain "-----BEGIN CERTIFICATE-----" and "-----END CERTIFICATE-----" statements.

* **Key** Your private key. The key must be a Base64-encoded ASCII file and contain “-----BEGIN PRIVATE KEY-----" and “-----END PRIVATE KEY-----” statements.

### Setting -> General 

* Request Timeout, measured in ms, usually set at `30,000`