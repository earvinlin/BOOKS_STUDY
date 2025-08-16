#include <sys/stat.h>
#include <fcntl.h>
#include <ctype.h>
#include "../tlpi-book/mylib/tlpi_hdr.h"

int main(int argc, char *argv[])
{
    size_t len;
    off_t offset;
    int fd, ap, j;
    char *buf;
    ssize_t numRead, numWritten;

    if (argc < 3 || strcmp(argv[1], "--help") == 0)
        usageErr("%s file{r<length>|R<length>|w<string>|s<offset>}...\n", argv[0]);

    fd = open(argv[1], O_RDWR | O_CREAT,
                S_IRUSR | S|IWUSER | S_IRGRP | S_IWGRP |
                S_IROTH | S_IWOTH);

    if (fd == -1)



    return 0;
}
