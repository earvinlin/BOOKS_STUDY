#include <dirent.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

/* List all files in directory 'dirPath' */
// dirpath → 要列出的目錄路徑
static void listFiles(const char *dirpath) {
    DIR *dirp;
    struct dirent *dp;
    Boolean isCurrent; /* True if 'dirpath' is "." */
    isCurrent = strcmp(dirpath, ".") == 0;
    dirp = opendir(dirpath); // 開啟目錄
    if (dirp == NULL) {
        errMsg("opendir failed on '%s'", dirpath);
        return;
    }

    /* For each entry in this directory, print directory + filename */
    for (;;) {
        errno = 0; /* To distinguish error from end-of-directory */
        dp = readdir(dirp); // 讀取目錄項目
        if (dp == NULL)
            break;

        // 過濾掉 "." 與 ".."
        if (strcmp(dp->d_name, ".") == 0 || strcmp(dp->d_name, "..") == 0)
            continue; /* Skip . and .. */
        // 印出檔名，若不是當前目錄，會加上路徑前綴
        if (!isCurrent)
            printf("%s/", dirpath);
        printf("%s\n", dp->d_name);
    }

    if (errno != 0)
        errExit("readdir");
    // 關閉目錄
    if (closedir(dirp) == -1)
        errMsg("closedir");
}

int main(int argc, char *argv[])
{
    if (argc > 1 && strcmp(argv[1], "--help") == 0)
        usageErr("%s [dir...]\n", argv[0]);
    
    // 如果沒有參數 → 列出目前目錄 "."
    if (argc == 1) /* No arguments - use current directory */
        listFiles(".");
    else 
    // 如果有參數 → 逐一列出每個指定目錄的檔案
        for (argv++; *argv; argv++)
            listFiles(*argv);
    
    exit(EXIT_SUCCESS);
}
