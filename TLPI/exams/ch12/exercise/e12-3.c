/*
    設計一個程式，將已開啟特定路徑名稱的行程列出，可透過檢測每個 /proc/PID/fd/* 符號連結
    的內容取得，會需要使用巢狀迴圈與 readdir（3），來掃描全部的/proc/PID 目錄，接著是在
    每個 /proc/PID 目錄中的每個/proc/PID/fd/ 條目內容。為了讀取/proc/PID/fd/n符號連
    結的內容，會需要使用18.5節所述的 readlinko）。

    解題思維與實作步驟
	1. 外層迴圈：遍歷 /proc 目錄
        • 使用 opendir("/proc") 與 readdir() 逐一讀取 /proc 底下的項目。
        • 檢查項目名稱是否全由數字組成（代表這是一個行程的 PID 目錄）。
	2. 內層迴圈：遍歷 /proc/[PID]/fd 目錄
        • 對於每個 PID 目錄，開啟 /proc/[PID]/fd 目錄。
        • 使用 readdir() 逐一讀取該行程開啟的所有檔案描述符（n）。
	3. 讀取符號連結（Symbolic Link）
        • /proc/[PID]/fd/n 都是指向實際檔案的符號連結。
        • 使用 readlink() 取得該符號連結指向的真實目標路徑。
	4. 比對與輸出
        • 比對 readlink() 讀出來的路徑是否與使用者輸入的「特定路徑名稱」相同。
        • 若相同，則印出該行程的 PID。
    
    Compile Cmd : gcc e12-3.c -o e12-3_arm
                  gcc e12-3.c -o e12-3_arm -Wno-format-truncation
    Exec Example :
        (1) 基本測試：尋找誰開啟了某個特定檔案
            sudo ./e12-3_arm /var/log/syslog
        (2) 測試目前終端機（TTY）設備
            # 1. 先用 tty 指令查詢當前終端機路徑
            tty
            # 2. 假設輸出為 /dev/pts/0，將其作為參數傳入
            sudo ./e12-3_arm /dev/pts/0
        (3) 測試共用函式庫或系統動態庫
            sudo ./e12-3_arm /lib/x86_64-linux-gnu/libc.so.6
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <dirent.h>
#include <unistd.h>
#include <limits.h>

// 檢查字串是否全為數字（用於判斷 PID）
int is_numeric(const char *str) {
    while (*str) {
        if (!isdigit(*str)) return 0;
        str++;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <target_file_path>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // 取得目標檔案的絕對路徑
    char target_path[PATH_MAX];
    if (realpath(argv[1], target_path) == NULL) {
        perror("realpath");
        exit(EXIT_FAILURE);
    }

    // 1. 開啟 /proc 目錄
    DIR *proc_dir = opendir("/proc");
    if (!proc_dir) {
        perror("opendir /proc");
        exit(EXIT_FAILURE);
    }

    struct dirent *proc_entry;

    // 外層迴圈：遍歷 /proc 下的所有項目
    while ((proc_entry = readdir(proc_dir)) != NULL) {
        // 只處理 PID 目錄（純數字）
        if (!is_numeric(proc_entry->d_name)) {
            continue;
        }

        char fd_dir_path[PATH_MAX];
        snprintf(fd_dir_path, sizeof(fd_dir_path), "/proc/%s/fd", proc_entry->d_name);

        // 開啟 /proc/[PID]/fd 目錄
        DIR *fd_dir = opendir(fd_dir_path);
        if (!fd_dir) {
            // 可能因為權限不足或行程已結束而失敗，直接跳過
            continue;
        }

        struct dirent *fd_entry;

        // 內層迴圈：遍歷 /proc/[PID]/fd 下的所有條目
        while ((fd_entry = readdir(fd_dir)) != NULL) {
            // 跳過 . 和 ..
            if (strcmp(fd_entry->d_name, ".") == 0 || strcmp(fd_entry->d_name, "..") == 0) {
                continue;
            }

            /*
                編譯器在進行靜態分析時，發現你的目的地陣列 symlink_path 大小為 4096（即 PATH_MAX），而來源字
                串 fd_dir_path（最大 4096）加上斜線 / 與 fd_entry->d_name（最大 255）的最糟情況組合，最大
                長度可能達到 4352 位元組。GCC 提醒你格式化後的結果可能會 被截斷（Truncated）。
            */
//            char symlink_path[PATH_MAX];
            char symlink_path[PATH_MAX + 256];
            snprintf(symlink_path, sizeof(symlink_path), "%s/%s", fd_dir_path, fd_entry->d_name);

            char link_target[PATH_MAX];
            // 使用 readlink() 讀取符號連結內容
            ssize_t len = readlink(symlink_path, link_target, sizeof(link_target) - 1);
            if (len != -1) {
                link_target[len] = '\0'; // 補上字串結尾標誌 '\0'

                // 比對路徑是否與目標相同
                if (strcmp(link_target, target_path) == 0) {
                    printf("PID %s HAS OPENED %s (fd: %s)\n", 
                           proc_entry->d_name, target_path, fd_entry->d_name);
                    break; // 該行程已確定開啟此檔案，可直接跳出內層迴圈
                }
            }
        }
        closedir(fd_dir);
    }

    closedir(proc_dir);
    return 0;
}
