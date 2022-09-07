#include<stdio.h>
#include<sys/types.h>
#include<sys/socket.h>
#include <errno.h>
#include <arpa/inet.h>
#include <string.h>

#define BUFFER_SIZE 1024

int main(int argc, char **argv) {
	int fd = socket(AF_INET, SOCK_STREAM, 0);
        printf("fd:%d\n", fd);

	struct sockaddr_in servaddr;
        bzero(&servaddr, sizeof(servaddr));
	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
	servaddr.sin_port = htons(8080);
	int ret = bind(fd, (struct sockaddr*)&servaddr, sizeof(servaddr));
	if (ret < 0) {
		printf("serv bind fail:%d\n", errno);
		return 0;
	}
	printf("serv bind success.\n");

	ret = listen(fd, 200);
	if (ret < 0) {
		printf("serv listen fail:%d\n", errno);
		return 0;
	}
	printf("serv listen success.\n");

	while(1){
		struct sockaddr_in cliaddr;
		socklen_t caddr_len = sizeof(cliaddr);
		int cfd = accept(fd, (struct sockaddr*)&cliaddr, &caddr_len);
		if(-1 == cfd) {
			printf("accept fail:%d\n", errno);
			return 0;
		}
		printf("client connect success:%d\n", cfd);
		char buffer[BUFFER_SIZE];

		while(1) {
			memset(buffer, 0, sizeof(buffer));
			int recvbytes = recv(cfd, buffer, sizeof(buffer),0);
			if(recvbytes == 0) {
				printf("client is disconnect.\n");
				break;
			}
			if(recvbytes < 0) {
				printf("recv err:%d.\n", errno);
				continue;
			}
			printf("client->server:%s\n", buffer);
        		send(cfd, buffer, recvbytes, 0);
		}
		close(cfd);
	}
	close(fd);
	return 0;
}