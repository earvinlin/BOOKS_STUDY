#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    if (execl("/sbin/ifconfig2", "/sbin/ifconfig2", NULL) == -1) {
        fprintf(stderr, "Not found in //sbin//ifconfig, err-cd: %s\n\n", strerror(errno));
        if (execlp("ipconfig", "ipconfig", NULL) == -1) {
            fprintf(stderr, "Cannot run ipconfig: %s\n", strerror(errno));
            return -1;
        }
    }

    return 0;
}
