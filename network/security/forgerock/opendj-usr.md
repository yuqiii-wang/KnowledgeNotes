# Opendj-usr

Run this for USR Store Creation:
```bash
./ssoadm create-datastore -e realm -t LDAPv3ForOpenDS -D config.conf
```
where `config.conf` is ldap usr schema shown as below

### LDAP Schema for **user** opendj

A typical DS USR Schema:

https://backstage.forgerock.com/docs/ds/7/schemaref/index.html#preface


Referenced from 

https://wikis.forgerock.org/confluence/display/openam/Assigned+OIDs+for+AttributeTypes
https://wikis.forgerock.org/confluence/display/openam/Other+OIDs+Used+By+OpenAM
https://wikis.forgerock.org/confluence/display/openam/Assigned+OIDs+for+ObjectClasses

```bash
( 2.16.840.1.113730.3.1.55
 NAME 'aci'
 DESC 'Sun ONE defined access control information attribute type'
 SYNTAX 1.3.6.1.4.1.1466.115.121.1.26
 USAGE directoryOperation
 X-ORIGIN 'Sun ONE Directory Server' )
```

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

( 2.16.840.1.113730.3.2.130 
NAME 'inetuser' 
DESC 'Auxiliary class which has to be present in an entry for delivery of subscriber services' 
SUP top 
AUXILIARY 
MAY ( uid $ inetUserStatus $ inetUserHTTPURL $ userPassword $ memberof ) 
X-ORIGIN 'Nortel subscriber interoperability' )

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

```bash
( 1.3.6.1.4.1.42.2.27.9.2.76 
NAME 'sunFederationManagerDataStore' 
DESC 'FSUser provider OC' 
SUP top 
AUXILIARY 
MAY ( iplanet-am-user-federation-info-key $ iplanet-am-user-federation-info $ sunIdentityServerDiscoEntries) 
X-ORIGIN 'Sun Java System Identity Management' )
```

```bash
( 2.16.840.1.113730.3.2.176 
NAME 'iplanet-am-user-service' 
DESC 'User Service OC' 
SUP top 
AUXILIARY 
MAY ( iplanet-am-user-auth-modules $ iplanet-am-user-login-status $ iplanet-am-user-admin-start-dn $ iplanet-am-user-auth-config $ iplanet-am-user-alias-list $ iplanet-am-user-success-url $ iplanet-am-user-failure-url $ iplanet-am-user-password-reset-options $ iplanet-am-user-password-reset-question-answer $ iplanet-am-user-password-reset-force-reset $ sunIdentityMSISDNNumber ) 
X-ORIGIN 'Sun Java System Identity Management' )

( 2.16.840.1.113730.3.2.184 
NAME 'iplanet-am-managed-person' 
DESC 'Managed Person OC' 
SUP top 
AUXILIARY 
MAY ( iplanet-am-modifiable-by $ iplanet-am-static-group-dn $ iplanet-am-user-account-life ) 
X-ORIGIN 'Sun Java System Identity Management' )

( 1.3.6.1.4.1.42.2.27.9.2.23 
NAME 'iplanet-am-auth-configuration-service' 
DESC 'Authentication Configuration Service OC' 
SUP top 
AUXILIARY 
MAY ( iplanet-am-auth-configuration $ iplanet-am-auth-login-success-url $ iplanet-am-auth-login-failure-url $ iplanet-am-auth-post-login-process-class ) 
X-ORIGIN 'Sun Java System Identity Management' )

( 1.3.6.1.4.1.1466.101.120.142 
NAME 'iPlanetPreferences' 
AUXILIARY 
MAY ( preferredLanguage $ preferredLocale $ preferredTimeZone ) 
X-ORIGIN 'iPlanet' )
```

```bash
 1.3.6.1.4.1.1466.101.120.1433 
 NAME 'forgerock-am-dashboard-service' 
 AUXILIARY 
 MAY ( assignedDashboard ) 
 X-ORIGIN 'Forgerock' )
```

```bash
( 1.3.6.1.4.1.36733.2.2.1.100 
NAME ( 'coreTokenObject' ) 
DESC 'Serialised JSON object for Token' 
SYNTAX 1.3.6.1.4.1.1466.115.121.1.5  
SINGLE-VALUE 
X-ORIGIN 'ForgeRock OpenAM CTSv2' )

( 1.3.6.1.4.1.36733.2.2.1.101 
NAME ( 'coreTokenString01' ) 
DESC 'General mapped string field' 
SYNTAX 1.3.6.1.4.1.1466.115.121.1.15  
SINGLE-VALUE 
X-ORIGIN 'ForgeRock OpenAM CTSv2' )
```

```bash
( 1.3.6.1.4.1.36733.2.2.1.96 
NAME ( 'coreTokenId' ) 
DESC 'Token unique ID' 
SYNTAX 1.3.6.1.4.1.1466.115.121.1.15  
SINGLE-VALUE 
X-ORIGIN 'ForgeRock OpenAM CTSv2' )
```

```bash
( 1.3.6.1.4.1.36733.2.2.2.27 
NAME 'frCoreToken' 
DESC 'object containing ForgeRock Core Token' 
SUP top STRUCTURAL 
MUST ( coreTokenId $ coreTokenType ) 
MAY ( coreTokenExpirationDate $ coreTokenUserId $ coreTokenObject $ coreTokenString01 $ coreTokenString02 $ coreTokenString03 $ coreTokenString04 $ coreTokenString05 $ coreTokenString06 $ coreTokenString07 $ coreTokenString08 $ coreTokenString09 $ coreTokenString10 $ coreTokenString11 $ coreTokenString12 $ coreTokenString13 $ coreTokenString14 $ coreTokenString15 $ coreTokenInteger01 $ coreTokenInteger02 $ coreTokenInteger03 $ coreTokenInteger04 $ coreTokenInteger05 $ coreTokenInteger06 $ coreTokenInteger07 $ coreTokenInteger08 $ coreTokenInteger09 $ coreTokenInteger10 $ coreTokenDate01 $ coreTokenDate02 $ coreTokenDate03 $ coreTokenDate04 $ coreTokenDate05 ) 
X-ORIGIN 'ForgeRock OpenAM CTSv2' )
```

```bash
( 1.3.6.1.4.1.42.2.27.4.1.6
    NAME 'javaClassName'
    DESC 'Fully qualified name of distinguished Java class or
        interface'
    EQUALITY caseExactMatch
    SYNTAX 1.3.6.1.4.1.1466.115.121.1.15
    SINGLE-VALUE
)

( 1.3.6.1.4.1.42.2.27.4.1.7
    NAME 'javaCodebase'
    DESC 'URL(s) specifying the location of class definition'
    EQUALITY caseExactIA5Match
    SYNTAX 1.3.6.1.4.1.1466.115.121.1.26
)

( 1.3.6.1.4.1.42.2.27.4.2.4
    NAME 'javaObject'
    DESC 'Java object representation'
    SUP top
    ABSTRACT
    MUST ( javaClassName )
    MAY ( javaClassNames $
        javaCodebase $
        javaDoc $
        description )
)
```