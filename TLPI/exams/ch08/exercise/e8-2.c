/**
 * 使用 setpwent()、geipwent() 和 endpwent() 來實作 gepwnam()。
 * 
    gepwnam() : 透過「使用者帳號名稱」查詢帳號資料的函式。
    #include <sys/types.h>
    #include <pwd.h>
    struct passwd *getpwnam(const char *name);
    參數與回傳值
    • name：欲查詢的使用者帳號名稱（例如 "root" 或 "earvin"）。
    • 回傳值：
    • 成功：回傳指向 struct passwd 結構的指標。
    • 失敗：若找不到該使用者或發生錯誤，回傳 NULL（可透過 errno 檢查錯誤原因）。
    struct passwd 結構欄位
    • pw_name (char *)：使用者帳號名稱。
    • pw_passwd (char *)：加密密碼（通常為 "x"，實際密碼存在 /etc/shadow）。
    • pw_uid (uid_t)：使用者 ID (UID)。
    • pw_gid (gid_t)：主要群組 ID (GID)。
    • pw_gecos (char *)：使用者真實姓名或詳細資訊。
    • pw_dir (char *)：家目錄路徑。
    • pw_shell (char *)：預設 Shell 路徑。   
 */
#include <stdlib.h>
#include <errno.h>
#include <pwd.h> 
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

void get_myptnam(const char *username) {
    struct passwd *pw;

    setpwent();

    while ((pw = getpwent()) != NULL) {
        if (strcmp(pw->pw_name, username) == 0) {
            // 印出查詢結果
            printf("=== 使用者資訊(自已實作) ===\n");
            printf("帳號名稱 (pw_name) : %s\n", pw->pw_name);
            printf("使用者 ID (pw_uid)  : %u\n", pw->pw_uid);
            printf("主要群組 ID (pw_gid): %u\n", pw->pw_gid);
            printf("家目錄 (pw_dir)     : %s\n", pw->pw_dir);
            printf("預設 Shell (pw_shell): %s\n", pw->pw_shell);
            break;
        }
    }
    endpwent();
}

void call_getpwnam(const char *username) {
    // 透過帳號名稱查詢使用者資訊
    struct passwd *pw = getpwnam(username);

    // 錯誤與存在性檢查
    if (pw == NULL) {
        if (errno != 0) {
            perror("getpwnam 查詢發生錯誤");
        } else {
            printf("找不到使用者: %s\n", username);
        }
        exit(EXIT_FAILURE);
    }

    // 印出查詢結果
    printf("=== 使用者資訊(呼叫getpwent()) ===\n");
    printf("帳號名稱 (pw_name) : %s\n", pw->pw_name);
    printf("使用者 ID (pw_uid)  : %u\n", pw->pw_uid);
    printf("主要群組 ID (pw_gid): %u\n", pw->pw_gid);
    printf("家目錄 (pw_dir)     : %s\n", pw->pw_dir);
    printf("預設 Shell (pw_shell): %s\n", pw->pw_shell);
}

int main(int argc, char *argv[])
{
// 檢查命令列參數
    if (argc < 2) {
        printf("使用方式: %s <username>\n", argv[0]);
        return 1;
    }

    const char *username = argv[1];
    
    // 展示原函式呼叫結果
    call_getpwnam(username);

    printf("\n=======================================\n\n");

    // 依題意自訂呼叫結果
    get_myptnam(username);

    return 0;
}
