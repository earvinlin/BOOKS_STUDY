/**
 * gcc readv_test.c -o readv_test
 * ./readv_test
 */
#include <sys/uio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define HDR_SIZE 16
#define BODY_SIZE 42


int main(void) {
    int fd = open("e5-7.bin", O_RDONLY);
    if (fd == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    while (1) {
        char header[HDR_SIZE];
        char body[BODY_SIZE];
        memset(header, ' ', HDR_SIZE);
        memset(body, ' ', BODY_SIZE);
        struct iovec iov[2];

        iov[0].iov_base = header;
        iov[0].iov_len  = HDR_SIZE;
        iov[1].iov_base = body;
        iov[1].iov_len  = BODY_SIZE;

        ssize_t nread = readv(fd, iov, 2);
        
        if (nread == 0) {
            printf("EOF reached\n");
            break;
        }
        if (nread < 0) {
            perror("readv");
            break;
        }    
        if (nread < HDR_SIZE + BODY_SIZE) {
            printf("Partial record (n=%zd), stopping\n", nread);
            // 列印最後一項 
            printf("Read one record: header='%.*s', body='%.*s'\n",
               HDR_SIZE, header, BODY_SIZE, body);        
            break;
        } 

        printf("Read one record: header='%.*s', body='%.*s'\n",
               HDR_SIZE, header, BODY_SIZE, body);
    }

/*
    printf("Total bytes read: %zd\n", nread);
    //-- Hex Code --//
    printf("Header: ");
    for (int i = 0; i < HDR_SIZE; i++)
        printf("%02x ", (unsigned char)header[i]);
    printf("\n");

    printf("Body: ");
    for (int i = 0; i < BODY_SIZE; i++)
        printf("%02x ", (unsigned char)body[i]);
    printf("\n");
    //-- Char Code --//
    printf("Header(char): ");
    for (int i = 0; i < HDR_SIZE; i++)
        printf("%c ", (unsigned char)header[i]);
    printf("\n");

    printf("Body(char): ");
    for (int i = 0; i < BODY_SIZE; i++)
        printf("%c ", (unsigned char)body[i]);
    printf("\n");
*/
    close(fd);
    return 0;
}
