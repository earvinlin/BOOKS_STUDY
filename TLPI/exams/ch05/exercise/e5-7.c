/**
 * 使用 read()、write()與 malloc package 中的適當函式實作 readv() 及 writev()(7.1.2 節)。
 *
 * 【編譯指令】 
    gcc e5-7.c my_readv.o my_writev.o \
    -I/home/earvin/workspaces/GithubProjects/BOOKS_STUDY/TLPI/tlpi-book/mylib \
    -L/home/earvin/workspaces/GithubProjects/BOOKS_STUDY/TLPI/tlpi-book/mylib \
    -ltlpi -o e5-7_arm
 *
 */
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "ex5-7.h"
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

#define HDR_SIZE 16
#define BODY_SIZE 42

int main(int argc, char *argv[])
{
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

//        ssize_t nread = readv(fd, iov, 2);
        ssize_t nread = my_readv(fd, iov, 2);

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

    // 使用 my_writev() 寫到另一個檔案

    return 0;
}
