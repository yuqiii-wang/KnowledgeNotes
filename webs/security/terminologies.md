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
