#include <termios.h>
#include <ctype.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif
	
int main(int argc, char *argv[])
{
	struct termios tp;
	int intrChar;

	if (argc > 1 && strcmp(argv[1], "--help") == 0)
		usageErr("%s [intr-char]\n", argv[0]);

	/* Determine new INTR setting from command line */

	if (argc == 1) { /* Disable */ /* 情境 A：沒給參數 */
		// 查詢這個 FD 所對應的終端機是否支援「停用控制字元」的功能。
		// 如果支援，會回傳：_POSIX_VDISABLE 的實際值（通常是 -1）
		// 如果不支援，會回傳：-1 且 errno = EINVAL
		// 查詢當前系統的 _PC_VDISABLE 標記（在 Linux 上通常是數值 0 或 \0），並賦值給 intrChar
		intrChar = fpathconf(STDIN_FILENO, _PC_VDISABLE);
		if (intrChar == -1)
			errExit("Couldn't determine VDISABLE");
	} else if (isdigit((unsigned char) argv[1][0])) {	// /* 情境 B：給了數字 */
		intrChar = strtoul(argv[1], NULL, 0); /* Allows hex, octal */
	} else { /* Literal character */
		intrChar = argv[1][0];
	}

	/* Fetch current terminal settings, modify INTR character, and
		push changes back to the terminal driver */

	// 1. 讀取：把當前標準輸入(終端機)的設定，讀進 tp 結構體中
	if (tcgetattr(STDIN_FILENO, &tp) == -1)
		errExit("tcgetattr");

	// 2. 修改：把我們在階段一決定好的 intrChar，塞進 c_cc 陣列的 VINTR 格子裡
	tp.c_cc[VINTR] = intrChar;

	// 3. 寫回：將修改後的 tp 結構體更新回作業系統核心
	if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &tp) == -1)
		errExit("tcsetattr");
	/*
		控制參數 TCSAFLUSH：這個參數非常重要。它的意思是「等目前所有正在排隊輸出的資料
		都傳完之後，再變更設定；而且變更的同時，要把目前使用者已經打進去、但程式還沒讀取
		的輸入緩衝區資料全部清空（Flush）」。這能保證切換快捷鍵時的資料乾淨，防止使用者
		因為誤操作而卡住。
	*/

	exit(EXIT_SUCCESS);
}

