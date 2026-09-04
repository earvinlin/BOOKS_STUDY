/**
 *  == 未完成 20260904 ==
    使用 setgroups（）及函式庫函式從密碼檔、群組檔（參考8.4節）檢素資訊，以實作 initgroups（）。
    請記得，呼叫 setgroups（）的行程必須具有特權。
    【題目解讀與要求分析】
	1. initgroups(const char *user, gid_t group) 的作用：
        • 讀取系統的群組檔案（通常為 /etc/group）。
        • 找出指定使用者 user 所屬的所有附加群組（Supplementary Groups）。
        • 將傳入的 group（通常為該使用者的主要群組 GID）也一併加入清單中。
        • 呼叫系統呼叫 setgroups()，將這組完整的 GID 清單設定給當前行程（Process）。
	2. 核心檢索來源：
        • 密碼檔 (/etc/passwd)：可用於驗證使用者是否存在，或取得其主要 GID。
        • 群組檔 (/etc/group)：需要走訪裡面的每一筆紀錄，檢查 user 是否列在該群組的成員名單中。
	3. 權限限制：
        • 呼叫 setgroups() 需要超級使用者權限（root / CAP_SETGID），因此實作出的函式在執行時必須具備特權。
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <grp.h>
#include <unistd.h>
#include <limits.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

/*
    【關鍵步驟說明】
	1. setgrent() & getgrent()：
    系統提供的標準庫函式，專門用來逐行解析 /etc/group 檔案並回傳 struct group 結構，省去手動用 fopen 解析字串的麻煩。
	2. 容量動態擴充 (realloc)：
    因為無法預知該使用者加入了多少個群組，使用動態陣列可以防止記憶體溢位。
	3. setgroups(ngroups, groups)：
    這是最終呼叫的特權系統呼叫（System Call），把整理好的 GID 陣列寫入核心的行程控制塊（PCB）中。
*/
int my_initgroups(const char *user, gid_t group) {
    struct group *gr;
    gid_t *groups = NULL;
    int ngroups = 0;
    int max_groups = 16; // 初始陣列大小

    // 1. 動態分配記憶體以儲存 GID 列表
    groups = malloc(max_groups * sizeof(gid_t));
    if (groups == NULL) {
        return -1;
    }

    // 2. 首先將傳入的基底群組 (主要群組 GID) 加入列表中
    groups[ngroups++] = group;

    // 3. 開啟並重置群組檔案 (/etc/group) 讀取指標
    setgrent();

    // 4. 逐筆讀取群組檔案
    while ((gr = getgrent()) != NULL) {
        // 檢查該群組的成員名單 (gr_mem)
        for (char **member = gr->gr_mem; *member != NULL; member++) {
            if (strcmp(*member, user) == 0) {
                // 如果成員匹配且該 GID 還沒加過，加入列表中
                if (ngroups >= max_groups) {
                    max_groups *= 2;
                    gid_t *new_groups = realloc(groups, max_groups * sizeof(gid_t));
                    if (new_groups == NULL) {
                        free(groups);
                        endgrent();
                        return -1;
                    }
                    groups = new_groups;
                }
                groups[ngroups++] = gr->gr_gid;
                break; // 找到後即可跳出目前群組的成員檢查
            }
        }
    }

    // 5. 關閉群組檔案
    endgrent();

    // 6. 呼叫系統呼叫 setgroups() 設定當前行程的附加群組（需 root 權限）
    int result = setgroups(ngroups, groups);

    // 7. 釋放記憶體並返回結果
    free(groups);
    return result;
}

int main(int argc, char *argv[])
{

    return 0;
}

