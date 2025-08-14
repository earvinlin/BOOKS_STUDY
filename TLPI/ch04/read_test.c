#include <sys/stat.h>
#include <fcntl.h>
#include "../tlpi-book/mylib/tlpi_hdr.h"
#define MAX_READ 20

char buffer[MAX_READ];

int main(int argc, char *argv[])
{
    if (read(STDIN_FILENO, buffer, MAX_READ) == -1)
        errExit("read");
    printf("The input data was %s\n", buffer);

    return 0;
}