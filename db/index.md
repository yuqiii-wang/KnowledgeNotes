# Index

`SHOW INDEX FROM <tablename>;` can show defined indices in a table. 

## Key Explained

* `PRIMARY KEY (a)`: The partition key is a.
* `PRIMARY KEY (a, b)`: The partition key is a, the clustering key is b.
* `PRIMARY KEY (a, b, c)`: The partition key is a, the composite clustering key is (b, c).
* `PRIMARY KEY ((a, b), c)`: The composite partition key is (a, b), the clustering key is c.


## Foreign Key vs Primary Key

*PRIMARY KEY* is used to identify a row entry, that must contain *UNIQUE* values, and cannot contain NULL values.

A *FOREIGN KEY* is a field (or collection of fields) in one table, that refers to the PRIMARY KEY in another table.

*FOREIGN KEY* requires server to do a lookup in another table when change happens such as `UPDATE`, to locate row entry addresses of the another table.

```sql
CREATE TABLE Persons (
    ID INT NOT NULL,
    LastName VARCHAR(255) NOT NULL,
    FirstName VARCHAR(255),
    Age INT,
    PRIMARY KEY (ID)
); 

CREATE TABLE Orders (
    OrderID int NOT NULL,
    OrderNumber int NOT NULL,
    PersonID int,
    PRIMARY KEY (OrderID),
    FOREIGN KEY (PersonID) REFERENCES Persons(PersonID)
); 
```
