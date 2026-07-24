#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

#define MAX_READ 30

//char buffer[MAX_READ];

int main(int argc, char *argv[]) {
    char buffer[MAX_READ];
//    memset(buffer, 'A', sizeof(buffer));
//    printf("The initial content is : %s\n\n", buffer);
    for (int i = 0; i <sizeof(buffer); i++) {
        printf("%02x ", (unsigned char) buffer[i]);
    }
    printf("\n");

    int fd = open("readFileTest.txt", O_RDONLY);
    if (fd == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    ssize_t n = read(fd, buffer, MAX_READ - 1);
    if (n == -1) {
        perror("read");
        exit(EXIT_FAILURE);
    }
    buffer[MAX_READ-2] = '*';
    buffer[MAX_READ-1] = '\0';

    printf("File content:\n%s\n", buffer);
/*
    // Wrong Writing
    if (read(STDERR_FILENO, buffer, MAX_READ) == -1)
        errExit("read");
    printf("The input data was: %s\n", buffer);

    // Correct Writing
    ssize_t n = read(STDERR_FILENO, buffer, MAX_READ - 1);
    if (n == -1)
        errExit("read");
    
//    buffer[n] = '\0';
    buffer[MAX_READ-2] = '*';
    buffer[MAX_READ-1] = '\0';
*/
    return 0;
}
