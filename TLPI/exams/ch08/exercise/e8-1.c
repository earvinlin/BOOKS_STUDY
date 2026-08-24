/**
 * 執行下列程式碼時，會顯示兩個不同使用者ID 的使用者名稱，我們發現程式將相同的使用者名稱顯示兩次。請問為什麼？
 * printf (%s %s", getpwuid(uid1)->pw_name, getpwuid(uid2)->pw_name);
 * 
    這行程式碼 printf("%s %s", getpwuid(uid1)->pw_name, getpwuid(uid2)->pw_name); 在 C 語言
    中包含嚴重的邏輯陷阱與崩潰風險。
    雖然語法上看起來是要一次印出 uid1 與 uid2 的使用者名稱，但在實際執行時幾乎無法得到正確結果，甚至可能
    導致程式直接當機（Segmentation Fault）。
    核心問題與風險分析
    • 靜態記憶體被覆蓋（最主要的問題）：
      getpwuid() 回傳的是指向系統內部同一塊靜態記憶體區塊（Static Buffer）的指標。
    當你在同一個 printf 呼叫中執行兩次 getpwuid() 時，第二次呼叫傳回的結果會直接覆蓋掉第一次呼叫寫入的資
    料。最終印出來的結果，通常會是兩個一模一樣的名稱（均為 uid2 或 uid1 的名稱）。
    • 參數求值順序未定義（Undefined Order of Evaluation）：
      C 語言標準並未規定函式參數的計算順序（編譯器可以自由選擇從左到右，或從右到左計算）。你無法預測 
      getpwuid(uid1) 先執行還是 getpwuid(uid2) 先執行，這讓程式的行為變得不可預測。
    • 空指標解引用崩潰（Null Pointer Dereference）：
      若 uid1 或 uid2 在系統中不存在，getpwuid() 會回傳 NULL。這時直接存取 ->pw_name 會觸發記憶體存取
      違規（Segmentation Fault），直接導致程式崩潰。
    結論：
    • 單一行內切勿多次呼叫回傳靜態指標的函式（如 getpwuid、getpwent、strtok、ctime 等）。
    • 在多執行緒環境下，請改用 Thread-safe 的版本 getpwuid_r()。
 *
 */
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <pwd.h>
#include <stdlib.h>
#include <errno.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

// 正確且安全的改寫方式
// 安全的作法是分開呼叫、檢查空指標，並將字串複製出來後再印出：
void safe_print_users(uid_t uid1, uid_t uid2) {
    char name1[64] = "Unknown";
    char name2[64] = "Unknown";

    // 1. 查詢 uid1 並安全複製
    struct passwd *pw1 = getpwuid(uid1);
    if (pw1 != NULL) {
        strncpy(name1, pw1->pw_name, sizeof(name1) - 1);
    }

    // 2. 查詢 uid2 並安全複製
    struct passwd *pw2 = getpwuid(uid2);
    if (pw2 != NULL) {
        strncpy(name2, pw2->pw_name, sizeof(name2) - 1);
    }

    // 3. 安全印出
    printf("%s %s\n", name1, name2);
}

// 安全將字串轉為 uid_t
uid_t get_user_id(char *user_id) {
    char *endptr;
    errno = 0;

    // 將字串依十進位轉換為 unsigned long
    unsigned long parsed_val = strtoul(user_id, &endptr, 10);

    // 錯誤檢查：是否有非數字字串、或發生溢位
    if (errno != 0 || *endptr != '\0' || endptr == user_id) {
        fprintf(stderr, "錯誤：無效的 UID 格式 '%s'\n", user_id);
        exit(EXIT_FAILURE);
    }

    // 安全轉換為 uid_t
    uid_t user_id1 = (uid_t)parsed_val;
    printf("成功解析的 UID: %u\n", user_id1);
    
    return user_id1;
}

int main(int argc, char *argv[])
{
    if (argc < 3 ) {
        printf("Syntax: %s user_id1 user_id2\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    
    uid_t user_id1 = get_user_id(argv[1]);
    uid_t user_id2 = get_user_id(argv[2]);
    // root : 0 ; earvin : 1000 ; nobody : 65534
    // ./e8-1_arm 1000 0
    safe_print_users(user_id1, user_id2);

    return 0;
}
