#include <sys/uio.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    int fd = open("data.txt", O_RDONLY);
    if (fd == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    char buf1[10], buf2[20], buf3[30];
    struct iovec iov[3] = {
        { .iov_base = buf1, .iov_len = sizeof(buf1) },
        { .iov_base = buf2, .iov_len = sizeof(buf2) },
        { .iov_base = buf3, .iov_len = sizeof(buf3) }
    };

    ssize_t nread = readv(fd, iov, 3);
    if (nread == -1) {
        perror("readv");
        exit(EXIT_FAILURE);
    }

    printf("Read %zd bytes\n", nread);
    close(fd);
    return 0;
}