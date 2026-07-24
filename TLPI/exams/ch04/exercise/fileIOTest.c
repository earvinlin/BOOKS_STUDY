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
    char buffer[BUF_SIZE];
    ssize_t bytes_read;
    
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
    if (argc < 3) {
        char *prog_name = basename(argv[0]);    // 使用 basename 取得不帶路徑的純檔名
        printf("parameter error, program is %s\n", argv[0]);
        printf("parameter error, program is %s\n", prog_name);

        char *s = argv[0];
        printf("=== %s ===\n", getProgName(s));
        exit(1);
    }

    char * op = argv[2];
    long sparseFileSize = 0;
    char * writeStr;
//    size_t len;
    off_t offset;

    switch (*op) {
        case 'r':
        case 'R':
            fd = open(argv[1], O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
            if (fd == -1) {
                perror("open fail! (for read)");
                return 1;
            }
            printf("成功開啟檔案，檔案描述符(FD)為：%d\n", fd);
            
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

            while ((bytes_read = read(STDIN_FILENO, buffer, BUF_SIZE)) > 0) {
                if (write(fd, buffer, bytes_read) != bytes_read) {
                    perror("write 寫入異常");
                    close(fd);
                    return 1;
                }
            }
            if (bytes_read == -1) {
                perror("read 讀取錯誤");
            }
            close(fd);
            printf("\n資料已成功寫入 %s！\n", argv[1]);
            break;
        
        case 'h' :
        case 'H' :
            sparseFileSize = atol(argv[3]);
            writeStr = argv[4];

            fd = open(argv[1], O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
            if (fd == -1) {
                perror("open fail!");
                return 1;
            }
            printf("成功建立檔案，檔案描述符(FD)為：%d\n", fd);     
            
            int numWritten = write(fd, writeStr, strlen(writeStr));
            if (numWritten == -1) {
                errExit("write");
                close(fd);        
                return 1;        
            }
            printf("%s: wrote %ld bytes\n", writeStr, (long) numWritten);

            // generate hold file
//            offset = getLong(&argv[ap][1], GN_ANY_BASE, argv[ap]);
//                if (lseek(fd, offset, SEEK_SET) == -1)
                if (lseek(fd, sparseFileSize, SEEK_SET) == -1)
                    errExit("lseek");
                printf("%s: seek succeeded\n", argv[3]);

            // 一定要寫入至少1byte資料，才會產生空洞檔案
            if (write(fd, "H", 1) != 1) {
                perror("write");
                exit(EXIT_FAILURE);
            }

            close(fd);
            break;

        default:
            printf("Unknown command line : %c\n", *op);
            break;
    }

    return 0;
}