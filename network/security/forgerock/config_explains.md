# ForgeRock config expalains

### Find param/config name
Use below cmd to extract an full existing config datasheet of an openam.
```bash
sudo ./ssoadm export-svc-cfg --admin amadmin -f "path/to/password/file" -e realm_name -o output_file_name.txt
```
Find required param name and default value, and see by which `ssoadm` argument to add them (such as `set-svc-attrs`, `set-realm-attrs`)


## Somep Params

* url path recognition
```conf
base-url-source=FORWARDED_HEADER
base-url-context-path=/reverseproxy/api/v1
```
Only permit one certain url path (a unique point of access) to invoke openam's services.

* host recognition
```conf
sunOrganizationAliases=nginx-proxy.examplecompany.com
com.sun.identity.serverfqdnMap[nginx-proxy.examplecompany.com]=nginx-proxy.examplecompany.com
```
Works for JWT's aud recognition.

* add scopes
```conf
forgerock-oauth2-provider-supported-scopes=openid
```
Works for JWT's claimed scopes