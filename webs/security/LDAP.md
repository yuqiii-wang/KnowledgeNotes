# LDAP

## Introduction

**LDAP** (lightweight directory access protocol) is a communication protocol that defines the methods in which a directory service can be accessed.

LDAP terminologies:
* Directory: LDAP database
* Entry: a single data unit/record in a directory
* Attribute: detail of an entry
* Matching Rule: search criteria for matching entries in a directory
* Object Class: a group of attributes defining a directory
* Schema: a packaged concept of attributes, object classes and matching rules
* LDIF: a plain text of LDAP entries (u can import `xxx.ldif` into an LDAP server, similar to importing `xxx.sql` to a SQL database), specified in RFC 2849
* DN: Distinguished Name, an entry unique identifier, collectively consisted of RDNs (Relative Distinguished Names) such as CN (Canonical Name), OU (Organization Unit), etc.

## Schema Example and Explain

Define a globally unique Object Identifier (OID). To obtain a registered OID at no cost, apply for an OID under the Internet Assigned Numbers Authority (IANA) maintained Private Enterprise arc. Any private enterprise (organization) may request an OID to be assigned under this arc.

Below are commonly used already registered OIDs by IANA.

**Table: Example OID hierarchy**

| OID | Assignment |
|-----|------------|
| 1.1 | Organization's OID |
| 1.1.1 | SNMP Elements |
| 1.1.2 | LDAP Elements |
| 1.1.2.1 | AttributeTypes |
| 1.1.2.1.1 | myAttribute |
| 1.1.2.2 | ObjectClasses |
| 1.1.2.2.1 | myObjectClass |

Then, assign definitions to an OID.

Below is an already registered OID definition that defines a `name` of a new OID `2.5.4.41` that, 
1. `1.3.6.1.4.1.1466.115.121.1.15` is an OID defining directoryString (UTF-8 encoded Unicode) syntax with specified length of 32768
2. matching rules include `case insensitive, space insensitive`
3. a `DESC` description for this attribute
```bash
attributeType ( 2.5.4.41 NAME 'name'
        DESC 'name(s) associated with the object'
        EQUALITY caseIgnoreMatch
        SUBSTR caseIgnoreSubstringsMatch
        SYNTAX 1.3.6.1.4.1.1466.115.121.1.15{32768} )
```

Below are commonly used syntax and match rules

**Table: Commonly Used Syntaxes**
| Name | OID | Description |
|---|---|---|
| boolean | 1.3.6.1.4.1.1466.115.121.1.7 | boolean value |
| distinguishedName | 1.3.6.1.4.1.1466.115.121.1.12	 | DN |
| directoryString | 1.3.6.1.4.1.1466.115.121.1.15 | UTF-8 string |

**Table : Commonly Used Matching Rules**
| Name | Type | Description |
|---|---|---|
| booleanMatch | equality | boolean |
| caseIgnoreMatch | equality | case insensitive, space insensitive |
| caseIgnoreOrderingMatch | ordering | case insensitive, space insensitive |

Define a class object, similar to defining an attribute, 
1. assign an OID to this objectclass with a `NAME`
2. inherited from `top` and `STRUCTURAL` object class
3.  a `DESC` description for this object class
4. contain `attribute1` and `attribute2` attribute definitions

```bash
objectclass ( 1.3.6.1.4.1.15490.2.2 NAME 'objectclass1'
                SUP top STRUCTURAL
                DESC 'an object class'
                MUST (attribute1 $ attribute2 ) 
            )
```

**Table: Attributes for Object Class**
| Attribute Identifier | Attribute Value Description |
|---|---|
| NUMERICOID (mandatory) | Unique object identifier (OID) |
| NAME | Object class's name |
| DESC | Object class's description |
| OBSOLETE | "true" if obsolete; "false" or absent otherwise |
| SUP | Names of superior object classes from which this object class is derived |
| STRUCTURAL | "true" if object class is structural; "false" or absent otherwise |
| AUXILIARY | "true" if object class is auxiliary; "false" or absent otherwise. An AUXILIARY object class is one that does not define the core type of an entry, but defines additional characteristics of that entry. An LDAP Entry can contain zero or more AUXILIARY object classes. |
| MUST | List of type names of attributes that must be present |
| MAY | List of type names of attributes that may be present |
| X-ORIGIN | X-ORIGIN typically specifies the origin of the attribute which is often an RFC (Request For Comments) and represents a document series containing technical and organizational notes about the Internet. |

Other necessary explained:
1. `top` is an abstract object class that is the parent of every LDAP object class. It is the one that defines that every object in LDAP must have an `objectClass` attribute.
2. An `ObjectClass` defined for use in the `STRUCTURAL` specification of the `DIT` is termed a `STRUCTURAL ObjectClass`
3. `$` represents element `AND` operation 
4. `SUBSTR`: Substrings is a LDAP Filter Choices that can be used to identify entries that contain a value for a given attribute that matches a specified substring. For example, `(cn=ab*def*mno*stu*yz)` is a `SUBSTR` filter for `cn` whose sub-init str must contain `ab`, and sub-any for `def`, `mno`, `stu` and `yz`.   

## Object Class Example and Explained

This example looks at `inetOrgPerson`

The `inetOrgPerson` object class is a general purpose object class that holds attributes about people, defined already in `rfc2798` (https://datatracker.ietf.org/doc/html/rfc2798).

```bash
( 2.16.840.1.113730.3.2.2
    NAME 'inetOrgPerson'
    SUP organizationalPerson STRUCTURAL
    MAY (
        audio $ businessCategory $ carLicense $ departmentNumber $
        displayName $ employeeNumber $ employeeType $ givenName $
        homePhone $ homePostalAddress $ initials $ jpegPhoto $
        labeledURI $ mail $ manager $ mobile $ o $ pager $
        photo $ roomNumber $ secretary $ uid $ userCertificate $
        x500uniqueIdentifier $ preferredLanguage $
        userSMIMECertificate $ userPKCS12
    )
)
```

Inside `MAY` defining many attributes, such as `carLicense`, which is a plain UTF-8 string (defined in syntax `1.3.6.1.4.1.1466.115.121.1.15`) with matching rules disregarding case sensitivity.
```bash
( 2.16.840.1.113730.3.1.1 NAME 'carLicense'
    DESC 'vehicle license or registration plate'
    EQUALITY caseIgnoreMatch
    SUBSTR caseIgnoreSubstringsMatch
    SYNTAX 1.3.6.1.4.1.1466.115.121.1.15 
)
```

The aforementioned inherited/parent/super object class `organizationalPerson` defined in `rfc4519` (https://datatracker.ietf.org/doc/html/rfc4519).

```bash
( 2.5.6.7 NAME 'organizationalPerson'
    SUP person
    STRUCTURAL
    MAY ( title $ x121Address $ registeredAddress $
        destinationIndicator $ preferredDeliveryMethod $
        telexNumber $ teletexTerminalIdentifier $
        telephoneNumber $ internationalISDNNumber $
        facsimileTelephoneNumber $ street $ postOfficeBox $
        postalCode $ postalAddress $ physicalDeliveryOfficeName $
        ou $ st $ l 
    ) 
)
```

which is inherited from `person` defined in `rfc4519` (https://datatracker.ietf.org/doc/html/rfc4519). It reaches `top` root objectclass hence no more explaining required.

```bash
( 2.5.6.6 NAME 'person'
    SUP top STRUCTURAL
    MUST ( sn $
        cn 
    )
    MAY ( userPassword $
        telephoneNumber $
        seeAlso $ description 
    ) 
)
```

## Entry Example and Explained

Entries are added to an LDAP system as branches on trees called **Data Information Trees**, or DITs.

```
dn: cn=John Doe, ou=people, dc=example,dc=com
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

Attributes such as `givenName`, `telephoneNumber` and `mail` are defined in `ObjectClass`. For example, `telephoneNumber` is defined in `people` ObjectClass.

Here below is an example of `people` ObjectClass definition.
```bash
( 2.5.6.6 NAME 'people' 
    SUP top STRUCTURAL 
    MUST ( sn $ cn $ ou) 
    MAY ( userPassword $ telephoneNumber $ seeAlso $ description )
)
```
This indicates that the `people` object class is a `STRUCTURAL` class with `OID 2.5.6.6`. Entries with the people object class are required to include the `sn` (surname), `ou` (organizational unit) and `cn` (common name) attribute types, and may also include any or all of the userPassword, telephoneNumber, seeAlso, and description attribute types. And because the people object class inherits from the `top` object class, entries containing the people object class are required to also include the objectClass attribute type (which is declared as mandatory in the top object class). The people object class is not obsolete and does not have a description.

## LDAP Cmds

### purposes

A Reference:
https://dracones.ideosystem.com/work/talks/2016-04-soit-directory-services.pdf

* search
Find object in the DIT, also used for reading data

* add
Creating objects

* modify
Changing objects

* delete
Removing objects

* bind
User authentication

Directory server authenticates every incoming
connection using objects in the DIT, which means that **DN is used as user identifier**, and this action is termed `bind`

![ldap_based_authentication](imgs/ldap_based_authentication.png "ldap_based_authentication")

### cmds

* `ladpadd`:  adds entries to a directory
* `ldapcompare`: compares attributes
* `ldapdelete`: deletes entries from a directory
* `ldapmodify`: modifies entries in a directory
* `ldapsearch`: search LDAP directory entries

### search

```bash
ldapsearch [-b <baseDN>] [-s <scope>] <filter>
 [ <attrs> ]
```
examples: 
```bash
ldapsearch -b 'ou=people,o=nlight' '(objectclass=*)'

ldapsearch -b 'o=nlight' -s one '(ou=*)'

ldapsearch -b "dc=evolveum,dc=com" -s sub
 "(uid=semancik)"
```

commons options
```bash
-h <ldapServerHostName>
-p <ldapServerPort>
-D <bindDN>
```