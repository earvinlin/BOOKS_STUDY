/**
 * 
    使用 setgroups()及函式庫函式從密碼檔、群組檔（參考8.4節）檢素資訊，以實作 initgroups()。
    請記得，呼叫 setgroups()的行程必須具有特權。

    setgroups() 是 Linux / Unix 系統程式設計中的一個系統呼叫 (System Call)，主要用於設
    定目前行程(Process)的附加群組 ID(Supplementary Group IDs)清單。
    int setgroups(size_t size, const gid_t *list);
    參數與傳回值說明
    • size：list 陣列中的元素數量 (即要設定的群組數量)。不能超過系統規定的上限 NGROUPS_MAX 
            (通常在 <limits.h> 中定義，Linux 上通常為 65536，但在舊系統中可能較小)。
    • list：指向包含 gid_t 型態群組 ID 陣列的指標。
    • 傳回值：
    • 成功：傳回 0。
    • 失敗：傳回 -1，並會設定全局變數 errno 以指示錯誤原因。
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
#include <grp.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

int main(int argc, char *argv[])
{
    // 1. 先將 size 設為 0，取得附加群組總數
    int ngroups = getgroups(0, NULL);
    if (ngroups == -1) {
        perror("getgroups 取得數量失敗");
        return EXIT_FAILURE;
    }

    printf("目前行程共有 %d 個附加群組。\n", ngroups);

    if (ngroups > 0) {
        // 2. 動態分配足夠的記憶體空間
        gid_t *groups = malloc(ngroups * sizeof(gid_t));
        if (groups == NULL) {
            perror("記憶體分配失敗");
            return EXIT_FAILURE;
        }

        // 3. 再次呼叫以填入 GID
        ngroups = getgroups(ngroups, groups);
        if (ngroups == -1) {
            perror("getgroups 讀取清單失敗");
            free(groups);
            return EXIT_FAILURE;
        }

        // 4. 印出所有 GID
        printf("附加群組 GID 清單: ");
        for (int i = 0; i < ngroups; i++) {
            printf("%d ", groups[i]);
        }
        printf("\n");

        free(groups);
    }

    return EXIT_SUCCESS;
}

/*    
    gid_t groups[] = {1000, 1001};
    size_t ngroups = sizeof(groups) / sizeof(groups[0]);

    if (setgroups(ngroups, groups) != 0) {
        perror("setgroups 失敗");
        return EXIT_FAILURE;
    }
    printf("成功設定附加群組！\n");
    
    return 0;
*/

