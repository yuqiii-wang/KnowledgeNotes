# LDAP Indexing

An index maps a key to an ID list, which is the set of entry IDs for the entries that match that index key. 

* Equality or Value indexes are used to identify entries containing an attribute value that exactly matches a given assertion value, such as `cn=Babs Jensen`

* Presence Index are used to identify entries that contain at least one value for a given attribute, such as `uid`

* SubstringIndex are used to identify entries that contain an attribute value matching a given substring assertion, such as `cn=*derson`

