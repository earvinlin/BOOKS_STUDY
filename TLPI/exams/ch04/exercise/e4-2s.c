#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#define BUF_SIZE 4096

int main(int argc, char *argv[])
{
    int inFd, outFd;
    ssize_t numRead;
    char buf[BUF_SIZE];

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <source> <dest>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    inFd = open(argv[1], O_RDONLY);
    if (inFd == -1) {
        perror("open source");
        exit(EXIT_FAILURE);
    }

    outFd = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (outFd == -1) {
        perror("open dest");
        exit(EXIT_FAILURE);
    }

    while ((numRead = read(inFd, buf, BUF_SIZE)) > 0) {
        for (ssize_t i = 0; i < numRead; ) {
            /* 如果是 0 bytes，偵測連續空洞 */
            if (buf[i] == 0) {
                ssize_t holeStart = i;
                while (i < numRead && buf[i] == 0)
                    i++;
                off_t holeSize = i - holeStart;
                /* 用 lseek 跳過空洞，不寫 0 */
                if (lseek(outFd, holeSize, SEEK_CUR) == -1) {
                    perror("lseek");
                    exit(EXIT_FAILURE);
                }
            } else {
                /* 非 0 bytes，正常寫出 */
                ssize_t dataStart = i;
                while (i < numRead && buf[i] != 0)
                    i++;
                ssize_t dataSize = i - dataStart;
                if (write(outFd, &buf[dataStart], dataSize) != dataSize) {
                    perror("write");
                    exit(EXIT_FAILURE);
                }
            }
        }
    }

    if (numRead == -1) {
        perror("read");
        exit(EXIT_FAILURE);
    }

    close(inFd);
    close(outFd);

    return 0;
}
