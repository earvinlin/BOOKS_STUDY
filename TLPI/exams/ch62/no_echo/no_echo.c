#include <termios.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

#define BUF_SIZE 100
int main(int argc, char *argv[])
{
	struct termios tp, save;
	char buf[BUF_SIZE];

	/* Retrieve current terminal settings, turn echoing off */

	// 1. 讀取並備份目前的終端機設定
	// tcgetattr()：這個系統呼叫負責把目前標準輸入（STDIN_FILENO，即鍵盤）的終端機設定讀取
	// 出來，並存進 tp 變數中。
	if (tcgetattr(STDIN_FILENO, &tp) == -1)
		errExit("tcgetattr");

	// 這一步非常重要！在修改終端機設定前一定要先備份。否則要是程式結束後沒有恢復，使用者的終端
	// 機會陷入打字看不到的窘境。
	save = tp; /* So we can restore settings later */

	// 2. 透過位元運算關閉 ECHO 旗標
	// 這是 C 語言中標準的**「清除特定位元（Bitmask 清除）」**語法。c_lflag 控制的是終端機
	// 的本地模式（Local modes）。這行程式碼的意思是：不管其他設定如何，唯獨把 ECHO 這個位元
	// 清空為 0。
	tp.c_lflag &= ~ECHO; /* ECHO off, other bits unchanged */
	// 把修改後的 tp 結構體重新寫回終端機驅動程式，讓設定生效。
	// TCSAFLUSH：這是生效時機的參數。意思是「等目前所有輸出都傳送完畢，且把目前沒讀完的輸入快
	// 取全部清空（Flush）之後，才改變設定」。這樣可以確保修改安全，不會弄髒畫面。
	if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &tp) == -1)
		errExit("tcsetattr");


	/* Read some input and then display it back to the user */

	// 3. 測試輸入與輸出
	/*
		實際執行效果：畫面上印出 Enter text: 。這時你開始在鍵盤上打字（例如輸入 hello），螢
		幕上完全是一片空白，看不到任何字。
		當你按下 Enter 鍵後（因為預設依然是 ICANON 規範模式，所以要按 Enter 程式才拿得到資
		料），fgets() 成功讀取，接著觸發下一行，螢幕才會蹦出 Read: hello。
	*/
	printf("Enter text: ");
	fflush(stdout);	// 確保提示文字立刻印在螢幕上

	if (fgets(buf, BUF_SIZE, stdin) == NULL)
		printf("Got end-of-file/error on fgets()\n");
	else
		printf("\nRead: %s", buf);

	/* Restore original terminal settings */

	// 4. 恢復終端機的原始狀態
	if (tcsetattr(STDIN_FILENO, TCSANOW, &save) == -1)
		errExit("tcsetattr");

	exit(EXIT_SUCCESS);
}


