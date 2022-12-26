# Docker Mysql

```bash
docker run --name some-mysql -e MYSQL_ROOT_PASSWORD=my-secret-pw -d mysql
```

```yml
# Use root/example as user/password credentials
version: '3.1'

services:

  db:
    image: mysql
    command: --default-authentication-plugin=caching_sha2_password
    restart: always
    environment:
      MYSQL_ROOT_PASSWORD: P@ssw0rd

  adminer:
    image: adminer
    restart: always
    ports:
      - 8080:8080
  
  volume:
    mysql-vol:
```