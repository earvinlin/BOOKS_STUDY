#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    printf("Diners: %s\n", argv[2]);
//    printf("Juice: %s\n", getenv("JUICE"));
    printf("Path: %s\n", getenv("PathA"));

    return 0;
}

