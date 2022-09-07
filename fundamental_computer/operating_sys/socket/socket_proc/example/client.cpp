#include<stdio.h>
#include<sys/types.h>
#include<sys/socket.h>
#include <errno.h>
#include <arpa/inet.h>
#include <string.h>
#include <unistd.h>

#define BUFFER_SIZE 1024

int main(int argc, char **argv) {
	int fd = socket(AF_INET, SOCK_STREAM, 0);
        printf("fd:%d\n", fd);

	struct sockaddr_in servaddr;
        bzero(&servaddr, sizeof(servaddr));
	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");
	servaddr.sin_port = htons(8080);
	int ret = connect(fd, (struct sockaddr*)&servaddr, sizeof(servaddr));
	if (ret < 0) {
		printf("serv connect fail:%d\n", errno);
		return 0;
	}
	printf("serv connect success.\n");

	char sendbuf[BUFFER_SIZE] = "hello";
	char recvbuf[BUFFER_SIZE];
	while(1) {
		send(fd, sendbuf, strlen(sendbuf),0); 
		printf("client->server:%s\n", sendbuf);

		memset(&recvbuf, 0, sizeof(recvbuf));
		recv(fd, recvbuf, sizeof(recvbuf),0);
		printf("server->client:%s\n\n", recvbuf);
		sleep(3);
	}
	return 0;
}