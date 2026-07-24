#include <stdio.h>
#include <fcntl.h>   // open(), O_CREAT, O_WRONLY
#include <unistd.h>  // close()
#include <sys/stat.h> // S_IRUSR, S_IWUSR 權限巨集
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif
#include <libgen.h> // 使用 basename() 必須引入此標頭檔
#include <string.h> // 使用 strrchr()

#define MAX_READ 20
#define BUF_SIZE 30  // 每次最多讀取 128 位元組

char * getProgName(char *s) {
    // 尋找字串中最後一個 '/'
    char *progName = strrchr(s, '/');

    if (progName != NULL) {
        progName++; // 跳過 '/' 本身，指向檔名開頭
    } else {
        progName = s; // 如果路徑中沒有 '/'，直接使用 argv[0]
    }
//    printf("僅取程式名稱 : %s\n", progName);
    return progName;
}

int main(int argc, char *argv[]) {
    int fd, ap;
    char buffer[MAX_READ];

    /* 
        argv的資料結構本質上是指向陣列的指標
        ex: ./myprog -v input.txt 100
        argv[0] ""./myprog" 程式本身的名稱/路徑（永遠是第一個參數）
        argv[1] "-v"        使用者傳入的第 1 個參數
        argv[2] "input.txt" 使用者傳入的第 2 個參數
        argv[3] "100"       使用者傳入的第 3 個參數 (注意：類型依然是字串)
        argv[4] NULL        C 語言標準規定，陣列最後一格永遠是 NULL 指針
    */

    printf("The argc value is %d\n", argc);
    if (argc != 3) {
        // 使用 basename 取得不帶路徑的純檔名
        char *prog_name = basename(argv[0]);
        printf("parameter error, program is %s\n", argv[0]);
        printf("parameter error, program is %s\n", prog_name);

        char *s = argv[0];
        printf("=== %s ===\n", getProgName(s));
        exit(1);
    }

/*
    // 建立新檔案，並設定權限為 0644 (所有者可讀寫，其他人唯讀)
    int fd = open(argv[1], O_WRONLY | O_CREAT | O_TRUNC, 
            S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);

    if (fd == -1) {
        perror("open fail!");
        return 1;
    }
    printf("成功建立檔案，檔案描述符(FD)為：%d\n", fd);
*/

    char * op = argv[2];
    switch (*op) {
        case 'r':
        case 'R':
            fd = open(argv[1], O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
            if (fd == -1) {
                perror("open fail! (for read)");
                return 1;
            }
            printf("成功開啟檔案，檔案描述符(FD)為：%d\n", fd);

            char buffer[BUF_SIZE];
            ssize_t bytes_read;
            
            while ((bytes_read = read(fd, buffer, BUF_SIZE)) > 0) {
                // 將讀取到的資料寫入 Terminal (標準輸出 STDOUT_FILENO = 1)
                if (write(STDOUT_FILENO, buffer, bytes_read) != bytes_read) {
                    perror("write 寫入螢幕失敗");
                    close(fd);
                    return 1;
                }
            }
            if (bytes_read == -1) {
                perror("read 發生錯誤");
            } else {
                printf("\n=== 檔案讀取完畢 (EOF) ===\n");
            }
            close(fd);  // 關閉檔案
            break;

        case 'w':
        case 'W':
        /*
            1. Ctrl + D 機制： 在 Linux Terminal 中，Ctrl + D 會發送 EOF（End Of File）訊號，
               讓 getchar() 回傳 EOF（通常為 -1），或是讓 read() 回傳 0，進而結束 while 迴圈。
	        2. Terminal 行緩衝（Line Buffering）： 終端機預設是按下 Enter 鍵後才會將整列資料送入程
               式緩衝區中處理。
	        3. open() 旗標選擇： 在系統呼叫範例中，使用 O_WRONLY | O_CREAT | O_TRUNC 可以確保每次
               執行時，若檔案不存在會建立新檔，已存在則清空舊內容重新寫入。        
        */
            // 開啟/建立檔案，設定權限為 0644 (使用者可讀寫，其他人唯讀)
            fd = open(argv[1], O_WRONLY | O_CREAT | O_TRUNC, 
                        S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
            if (fd == -1) {
                perror("open fail!");
                return 1;
            }
            printf("成功建立檔案，檔案描述符(FD)為：%d\n", fd);     

            char buffer2[BUF_SIZE];
            ssize_t bytes_read2;

            while ((bytes_read2 = read(STDIN_FILENO, buffer2, BUF_SIZE)) > 0) {
                if (write(fd, buffer2, bytes_read2) != bytes_read2) {
                    perror("write 寫入異常");
                    close(fd);
                    return 1;
                }
            }
            if (bytes_read2 == -1) {
                perror("read 讀取錯誤");
            }
            close(fd);
            printf("\n資料已成功寫入 %s！\n", argv[1]);

            break;

        default:
            printf("Unknown command line : %c\n", *op);
            break;
    }
  

/*
    // 呼叫 write 寫入檔案
    char text[10] ="work work";
//    const char *text = "test test\n";
    size_t len = 10;
    ssize_t bytes_written = write(fd, text, len);

    // 防呆：檢查是否寫入成功且長度相符
    if (bytes_written == -1) {
        perror("write 到檔案失敗");
        close(fd);
        return 1;
    } else if ((size_t)bytes_written != len) {
        fprintf(stderr, "警告：未完整寫入資料！預期 %zu，實際寫入 %zd\n", len, bytes_written);
    } else {
        printf("成功寫入 %zd 位元組到 test_write.txt\n", bytes_written);
    }
*/

    return 0;
}