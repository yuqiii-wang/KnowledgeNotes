# Summary

* `int fd = socket(AF_INET, SOCK_STREAM, 0);`

* `int ret = bind(fd, (struct sockaddr*)&servaddr, sizeof(servaddr));`

* `ret = listen(fd, 200);`

* `int cfd = accept(fd, (struct sockaddr*)&cliaddr, &caddr_len);`

* `int ret = connect(fd, (struct sockaddr*)&servaddr, sizeof(servaddr));`