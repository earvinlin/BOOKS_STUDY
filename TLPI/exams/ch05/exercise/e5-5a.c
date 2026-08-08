/**
 * 寫個程式驗證複製檔案描述符會共用一個檔案偏移值及開啟檔案狀態旗標。
 */
#include <sys/stat.h>
#include <fcntl.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

// 查詢開啟檔案目前的檔案偏移值
off_t get_file_offset(int fd) {
    // 從「目前位置 (SEEK_CUR)」移動 0 位元組
    // lseek 會傳回當前的絕對偏移位置
    off_t current_offset = lseek(fd, 0, SEEK_CUR);

    if (current_offset == (off_t) -1) {
        perror("查詢檔案偏移值失敗 (lseek error)");
    }

    return current_offset;
}

// 查詢傳入之檔案描述符目前的狀態旗標
void print_file_flags(int fd) {
    int flags = fcntl(fd, F_GETFL);
    if (flags == -1) {
        perror("Function F_GETFL error!");
        return;
    }
    
    printf("檔案描述符 %d 的狀態旗標解析：\n", fd);

    /* --- 檢查存取模式 (Access Mode) ---
       注意：O_RDONLY(0), O_WRONLY(1), O_RDWR(2) 不是獨立的 Bit，
       必須用 O_ACCMODE 進行位元與（&）運算後才能比對！ */
    switch (flags & O_ACCMODE) {
        case O_RDONLY:
            printf("  - 存取模式: 唯讀 (O_RDONLY)\n");
            break;
        case O_WRONLY:
            printf("  - 存取模式: 唯寫 (O_WRONLY)\n");
            break;
        case O_RDWR:
            printf("  - 存取模式: 讀寫 (O_RDWR)\n");
            break;
        default:
            printf("  - 存取模式: 未知\n");
    }

    /* --- 檢查其他動態狀態旗標 (Status Flags) ---
       這些旗標是獨立的 Bit，直接使用 & 運算子檢查即可 */
    if (flags & O_APPEND) {
        printf("  - 設定了追加模式 (O_APPEND)\n");
    }
    if (flags & O_NONBLOCK) {
        printf("  - 設定了非阻塞模式 (O_NONBLOCK)\n");
    }
    if (flags & O_SYNC) {
        printf("  - 設定了同步寫入模式 (O_SYNC)\n");
    }
}

// 
void print_st_mode(int fd) {
    struct stat sb;
    char mode_str[11] = "----------"; // 預設 10 個位置 + '\0'

    // 1. 呼叫 fstat 取得檔案的 inode 屬性結構
    if (fstat(fd, &sb) == -1) {
        perror("fstat 失敗");
        return;
    }

    /* --- 1. 解析檔案類型 (File Type) --- */
    if (S_ISDIR(sb.st_mode))  mode_str[0] = 'd'; // 目錄
    else if (S_ISLNK(sb.st_mode)) mode_str[0] = 'l'; // 符號連結
    else if (S_ISCHR(sb.st_mode)) mode_str[0] = 'c'; // 字元裝置
    else if (S_ISBLK(sb.st_mode)) mode_str[0] = 'b'; // 區塊裝置
    else if (S_ISFIFO(sb.st_mode)) mode_str[0] = 'p'; // FIFO / Pipe
    else if (S_ISSOCK(sb.st_mode)) mode_str[0] = 's'; // Socket
    // 一般檔案 (S_ISREG) 保持為 '-'

    /* --- 2. 解析 User (擁有者) 權限 --- */
    if (sb.st_mode & S_IRUSR) mode_str[1] = 'r';
    if (sb.st_mode & S_IWUSR) mode_str[2] = 'w';
    if (sb.st_mode & S_IXUSR) mode_str[3] = 'x';

    /* --- 3. 解析 Group (群組) 權限 --- */
    if (sb.st_mode & S_IRGRP) mode_str[4] = 'r';
    if (sb.st_mode & S_IWGRP) mode_str[5] = 'w';
    if (sb.st_mode & S_IXGRP) mode_str[6] = 'x';

    /* --- 4. 解析 Others (其他人) 權限 --- */
    if (sb.st_mode & S_IROTH) mode_str[7] = 'r';
    if (sb.st_mode & S_IWOTH) mode_str[8] = 'w';
    if (sb.st_mode & S_IXOTH) mode_str[9] = 'x';

    /* --- 5. 解析特殊權限 (Set-UID, Set-GID, Sticky Bit) --- */
    if (sb.st_mode & S_ISUID) mode_str[3] = (sb.st_mode & S_IXUSR) ? 's' : 'S';
    if (sb.st_mode & S_ISGID) mode_str[6] = (sb.st_mode & S_IXGRP) ? 's' : 'S';
    if (sb.st_mode & S_ISVTX) mode_str[9] = (sb.st_mode & S_IXOTH) ? 't' : 'T';

    /* --- 印出顯示結果 --- */
    printf("ls -l 格式字串: %s\n", mode_str);
    printf("八進位數字表示: %o (全欄位: %o)\n", sb.st_mode & 07777, sb.st_mode);
    printf("十進位原始數值: %u\n", (unsigned int) sb.st_mode);
}


int main(int argc, char *argv[])
{
    int fd, old_fd,new_fd;
    int flag =  O_RDWR | O_CREAT;
    mode_t mode =  S_IRUSR | S_IWUSR;

    fd = open(argv[1], flag, mode);
    if (fd == -1)
        errExit("open");
    printf("fd open %d success.\n", old_fd);        

    if (write(fd, "a", 1) == -1) 
        errExit("write");
    if (write(fd, "\n", 1) == -1) 
        errExit("write");

    print_file_flags(fd);
    print_st_mode(fd);
    printf("Current cursor positon : %lld\n", (long long) get_file_offset(fd));

    exit(EXIT_SUCCESS);
}
