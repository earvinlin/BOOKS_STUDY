/**
 * 使用getenv()函式、putenv()函式，必要時可直接修改environ，來實作setenv()函式和
 * unsetenv(函式。此處的unsetenv()函式應檢查是否對環境變數進行了多次定義，如果是多
 * 次定義則將移除對該變數的全部定義(glibc版本的unsetenv()函式實作了這一功能)。
 */
//#include <sys/stat.h>
//#include <sys/types.h>
//#include <sys/uio.h>
//#include <fcntl.h>
//#include <unistd.h>
#include <stdlib.h>
//#include <string.h>
#include <errno.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif


void getenv_test() {
    /** char *getenv(const char *name);  **/
    // 取得 HOME 環境變數
    char *home = getenv("HOME");
    if (home != NULL) {
        printf("HOME path: %s\n", home);
    } else {
        printf("HOME variable is not set.\n");
    }

    // 查詢不存在的變數
    char *dummy = getenv("NON_EXISTENT_VAR");
    if (dummy == NULL) {
        printf("NON_EXISTENT_VAR does not exist.\n");
    }    
}

void setenv_test() {
    // 使用動態分配或字串字面值（需注意修改權限）
    char *env_str = malloc(32);
    sprintf(env_str, "MY_APP_LOG=%s", "enabled");

    if (putenv(env_str) != 0) {
        perror("putenv 失敗");
        free(env_str); // 釋放記憶體
    }

    // 驗證是否設定成功
    printf("MY_APP_LOG: %s\n", getenv("MY_APP_LOG")); // 印出: enabled
 
    /**
    注意：因為 putenv 拿走了 env_str 指標，此時不能馬上 free(env_str)！
    因為putenv()並沒有複製一份字串內容，而是直接把你的指標(記憶體位址)拿去當作系統環境變數
    的內容。如果呼叫putenv()後立刻執行free(env_str)，會導致系統環境變數指向一塊已被釋放
    的無效記憶體，造成未定義行為。
     */
}

int main(int argc, char *argv[])
{
    printf("=== Testing getenv() ===\n");
    getenv_test();
    printf("=== Testing setenv() ===\n");
    setenv_test();

    return 0;
}
