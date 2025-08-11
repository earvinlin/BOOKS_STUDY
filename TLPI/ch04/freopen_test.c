#include <stdio.h>

int main() {
    int a, b;
    /**
     * in.txt format :
     * 20 55
     * -8 37
     * 99 171
     * out.txt result :
     * 75
     * 29
     * 270
     */
    freopen("in.txt", "r", stdin);
    freopen("out.txt", "w", stdout);

    while (scanf("%d %d", &a, &b) != EOF)
        printf("%d\n", a + b);

    fclose(stdin);
    fclose(stdout);
    return 0;
}
