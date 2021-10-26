# Opendj CFG

To config an AM server, use the `configurator.jar` tool (for example, `openam-configurator-tool-14.1.3.4.jar` for AM 7) to configure a deployed AM server. The finished configs are stored on Opendj CFG.

Sample config file can be found here:

https://github.com/OpenIdentityPlatform/OpenAM/blob/master/openam-distribution/openam-distribution-ssoconfiguratortools/src/main/assembly/config/sampleconfiguration


### User Store
You can config user store (opendj-usr) to one of the followings:
* USERSTORE_TYPE

    1. `LDAPv3ForOpenDS`: ForgeRock 0penDJ or Sun OpenDS

    2. `LDAPv3ForAD`: Active Directory with host and port settings

    3. `LDAPv3ForADDC`: Active Directory with a Domain Name setting

    4. `LDAPv3ForADAM`: Active Directory Application Mode

    5. `LDAPv3ForODSEE`: Sun Java System Directory Server

    6. `LDAPv3ForTivoli`: IBM Tivoli Directory Server

### Cfg Store

The directory server where AM stores its configuration.
* DATA_STORE

    Set to Opendj-cfg itself

### AM Server

Config AM server URL:
```conf
SERVER_URL=https://openam.example.com:8443
```

## Example Config File

```conf
# Server properties, AM_ENC_KEY from first server
SERVER_URL=https://server2.example.com:8443
DEPLOYMENT_URI=/openam
BASE_DIR=$HOME/openam
locale=en_US
PLATFORM_LOCALE=en_US
AM_ENC_KEY=O6QWwHPO4os+zEz3Nqn/2daAYWyiFE32
ADMIN_PWD=change3me
AMLDAPUSERPASSWD=secret12
COOKIE_DOMAIN=openam.example.com
ACCEPT_LICENSES=true

# External configuration data store
DATA_STORE=dirServer
DIRECTORY_SSL=SSL
DIRECTORY_SERVER=opendj.example.com
DIRECTORY_PORT=1636
DIRECTORY_ADMIN_PORT=4444
DIRECTORY_JMX_PORT=1689
ROOT_SUFFIX=o=openam
DS_DIRMGRDN=uid=admin
DS_DIRMGRPASSWD=chang3me

# External DS-based user data store
USERSTORE_TYPE=LDAPv3ForOpenDS
USERSTORE_SSL=SSL
#USERSTORE_DOMAINNAME=ad.example.com
USERSTORE_HOST=opendj.example.com
USERSTORE_PORT=1636
USERSTORE_SUFFIX=dc=example,dc=com
USERSTORE_MGRDN=uid=admin
USERSTORE_PASSWD=secret12


# Site properties
LB_SITE_NAME=lb
LB_PRIMARY_URL=http://lb.example.com:80/openam
```