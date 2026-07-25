/*
 ex4-2 (不同的作業環境都要重新編譯才可執行，即使都是linux作業系統(debian, ubuntu)也相同)
 step1. 要先編譯 error_functions.c 為目的檔
        gcc -c /home/earvin/workspaces/GithubProjects/BOOKS_STUDY/TLPI/tlpi-book/mylib/error_functions.c -o error_functions.o
 step2. 再編譯ex4-2.c
        gcc ex4-2.c error_functions.o -I/home/earvin/workspaces/GithubProjects/BOOKS_STUDY/TLPI/tlpi-book/mylib -o ex4-2
*/
#include <sys/stat.h>
#include <fcntl.h>
#include "../tlpi-book/mylib/tlpi_hdr.h"

int main(int argc, char *argv[])
{
    /* Open existing file for reading */
    int fd0 = open("startup", O_RDONLY);
    if (fd0 == -1)
        errExit("open fd0");
    
    /* Open new or existing file for reading and writing, truncating to 
       zero bytes; file permissions read+write for owner, nothing for all
       others
     */
    int fd1 = open("myfile", O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd1 == -1)
        errExit("open fd1");
    
    /* Open new or existing file for writing; writes should allways 
       append to end of file
     */
    int fd2 = open("w.log", O_WRONLY | O_CREAT | O_APPEND, S_IRUSR | S_IWUSR);
    if (fd2 == -1)
        errExit("open fd2");

    return 0;
}