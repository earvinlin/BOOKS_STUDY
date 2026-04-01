#include <stdio.h>
#include <unistd.h>

int main() {
    uid_t ruid, euid, suid;

    if (getresuid(&ruid, &euid, &suid) == -1) {
        perror("getresuid");
        return 1;
    }

    printf("Real UID: %d\n", ruid);
    printf("Effective UID: %d\n", euid);
    printf("Saved UID: %d\n", suid);

    return 0;
}
