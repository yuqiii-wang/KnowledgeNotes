# Postgre SQL

## Quick Start

Start with a new db, schema and table.

```sql
CREATE DATABASE my_database;
CREATE SCHEMA my_schema;
CREATE TABLE my_schema.my_table (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    age INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Create a new user and grant necessary permissions.

```sql
CREATE USER my_user WITH PASSWORD 'my_secure_password';

-- Grant usage on the schema
GRANT USAGE ON SCHEMA my_schema TO my_user;

-- Grant permissions on the table
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA my_schema TO my_user;

-- Ensure future tables in the schema also get permissions
ALTER DEFAULT PRIVILEGES IN SCHEMA my_schema GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO my_user;
```

## Postgre SQL Knowledge

* The `public` schema

The `public` schema is the default schema in PostgreSQL.
Every `CREATE DATABASE <new_db>;` comes with a created `public` schema.

By default, all database users have access to the `public` schema and can create objects in it.
The default `public` schema is used to store management data of a database.

It is NOT recommended to use `public` schema, but `CREATE SCHEMA my_schema;`.
