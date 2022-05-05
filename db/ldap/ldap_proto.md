# LDAP Protocol


A client starts an LDAP session by connecting to an LDAP server, by default on TCP and UDP port 389, or on port 636 for LDAPS (LDAP over TLS/SSL.

1. StartTLS – use the LDAPv3 Transport Layer Security (TLS) extension for a secure connection
2. Bind – authenticate and specify LDAP protocol version
3. Query - Search, Compare, Add, Delete, Modify, etc.
4. Unbind – abandons any outstanding operations and closes the connection (not the inverse of Bind)

### Complex Query

To search on a given dn and two attributes `attr1` and `attr2` having NOT EQUAL AND and `attr3` and `attr4` having EQUAL OR:
```bash
./ldapsearch -h ${ldap_server_host} -p ${port} -D "${bindDN}" -W "${bindPassword}" -b ${baseDN} \
-s sub "(&(!(attr1=val1))(!(attr2=val2))(|(attr3=val3)(attr4=val4)))"
```