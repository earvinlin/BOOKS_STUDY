/**
 * 使用fcntl() 實作dup() 及 dup2()，並在有需要之處執行close()(你可以忽略dup2()
 * 及 fcntl() 實際上對於某些錯誤的例子會傳回不一樣的errno 值)。對於dup2()，記得要
 * 處理特殊的例子(在oldfd 等於newfd 之處)。在此例中，你您該檢查oldfd 是否為有效值，
 * 例如：檢查 fcntlfoldfd, F_GETFL） 是否成功。若 oldfd 不是有效值，那麼函式應該
 * 傳回 -1 並將 errno 設定為 EBADF。
 */
#include <sys/stat.h>
#include <fcntl.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

int main(int argc, char *argv[])
{

    exit(EXIT_SUCCESS);
}
