#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>


// 將 src 字串中從 start 位置起，複製 length 個字元到 dest 中
void substring(const char *src, int start, int length, char *dest) 
{
    int i;
    for (i = 0; i < length && src[start + i] != '\0'; i++) {
        dest[i] = src[start + i];
    }
    dest[i] = '\0';  // 別忘了結尾字元！
}

char* now()
{
    time_t t;
    time(&t);
    return asctime(localtime(&t));
}

int main()
{
    char comment[80];
    char cmd[120];

    fgets(comment, 80, stdin);
    sprintf(cmd, "echo '%s %s' >> reports.log", comment, now());
//    printf("cmd= %s\n", cmd);
    system(cmd);

//    printf("%zu\n", strlen(comment));

    return 0;
}


/** 
int main() {
    char str[] = "Hello, world!";
    char sub[20];

    substring(str, 7, 5, sub);  // 從第 7 個字元起（'w'），取 5 個字元
    printf("子字串是：%s\n", sub);  // 輸出：world

    return 0;
}
*/
