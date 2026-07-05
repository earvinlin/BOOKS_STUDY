#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    if (execlp("./coffee", "ban", NULL) == -1) {
        fprintf(stderr, "Can't create order: %s\n", strerror(errno));
        return -1;
    }

    return 0;
}
