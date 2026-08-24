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
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

struct passwd *get_myptnam(const char*name) {

}

int main(int argc, char *argv[])
{
    
    return 0;
}
