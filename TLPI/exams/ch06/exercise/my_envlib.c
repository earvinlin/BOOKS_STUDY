/**
 * gcc -Wall -Wextra -g -c my_envlib.c -o my_envlib.o
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "e6-3.h"

extern char **environ;

int my_setenv(const char *name, const char *value, int overwrite) {
    if (name == NULL || name[0] == '\0' || strchr(name, '=') != NULL) {
        return -1; // 無效的名稱
    }

    if (getenv(name) != NULL && !overwrite) {
        return 0; // 變數已存在且不覆蓋
    }

    // 格式化為 "NAME=VALUE"
    size_t len = strlen(name) + strlen(value) + 2;
    char *env_str = malloc(len);
    if (!env_str) return -1;
    
    snprintf(env_str, len, "%s=%s", name, value);

    // putenv 會直接將指標放入 environ 中
    return putenv(env_str); 
}

int my_unsetenv(const char *name) {
    if (name == NULL || name[0] == '\0' || strchr(name, '=') != NULL) {
        return -1;
    }

    size_t name_len = strlen(name);
    char **ep = environ;

    if (!ep) return 0;

    // 走訪並刪除所有符合名稱的項目
    while (*ep) {
        if (strncmp(*ep, name, name_len) == 0 && (*ep)[name_len] == '=') {
            // 找到相符項目，將後方的指標全部往前移動補齊
            char **next = ep;
            while (*next) {
                *next = *(next + 1);
                next++;
            }
            // 注意：這裡不移動 ep，繼續檢查新的 *ep 以防有重複定義
        } else {
            ep++;
        }
    }
    return 0;
}
