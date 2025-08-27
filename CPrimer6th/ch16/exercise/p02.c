// 調和平均數是將所有數值的倒數取算術平均後，再將結果取倒數
#include <stdio.h>
#include <stdlib.h>
#define H(X, Y) (2 / ((1/(X)) +(1/(Y))))

int main(int argc, char * argv[])
{
    if (argc < 3) {
        printf("parameter error!!!\n");
        exit(1);
    }
    float x = atof(argv[1]);
    float y = atof(argv[2]);
    printf("%.2f 和 %.2f 的調和數是 %.2f\n", x, y, H(x, y));

    return 0;
}
