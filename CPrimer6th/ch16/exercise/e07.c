#include <stdio.h>
#define PRN_ALL(X, Y, Z) printf("name: %s;  value: %i;  address: %p\n", X, Y, Z)


int main()
{
    char * name = "fop";
    int age = 23;
    PRN_ALL(name, age, &age);

    return 0;
}