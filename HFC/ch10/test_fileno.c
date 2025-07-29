#include <stdio.h>

int main() {
    FILE *fp = fopen("/etc/passwd", "r");
    if (!fp) {
        perror("fopen failed");
        return 1;
    }

    int fd = fileno(fp);
    printf("File descriptor: %d\n", fd);

    fclose(fp);
    return 0;
}
