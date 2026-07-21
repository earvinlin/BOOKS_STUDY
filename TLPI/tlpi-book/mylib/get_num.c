/*************************************************************************\
*                  Copyright (C) Michael Kerrisk, 2025.                   *
*                                                                         *
* This program is free software. You may use, modify, and redistribute it *
* under the terms of the GNU Lesser General Public License as published   *
* by the Free Software Foundation, either version 3 or (at your option)   *
* any later version. This program is distributed without any warranty.    *
* See the files COPYING.lgpl-v3 and COPYING.gpl-v3 for details.           *
\*************************************************************************/

/* Listing 3-6 */

/* get_num.c

   Functions to process numeric command-line arguments.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <errno.h>
#include "get_num.h"

/* Print a diagnostic message that contains a function name ('fname'),
   the value of a command-line argument ('arg'), the name of that
   command-line argument ('name'), and a diagnostic error message ('msg'). */
static void
gnFail(const char *fname, const char *msg, const char *arg, const char *name)
{
    fprintf(stderr, "%s error", fname);
    if (name != NULL)
        fprintf(stderr, " (in %s)", name);
    fprintf(stderr, ": %s\n", msg);
    if (arg != NULL && *arg != '\0')
        fprintf(stderr, "        offending text: %s\n", arg);

    exit(EXIT_FAILURE);
}

/* Convert a numeric command-line argument ('arg') into a long integer,
   returned as the function result. 'flags' is a bit mask of flags controlling
   how the conversion is done and what diagnostic checks are performed on the
   numeric result; see get_num.h for details.

   'fname' is the name of our caller, and 'name' is the name associated with
   the command-line argument 'arg'. 'fname' and 'name' are used to print a
   diagnostic message in case an error is detected when processing 'arg'. */

/*
    1. 參數與變數說明
    • fname(函式名稱)   : 呼叫此建構子時傳入的當前函式名（例如 main），用於錯誤發生時列印
                         具體是哪個地方出錯。
    • arg(待解析字串)   : 要轉換成數字的字串（例如 argv[1]）。
    • flags(控制旗標)   : 控制轉換行為的位元遮罩（Bitmask），例如進位制、正負號限制。
    • name(參數名稱)    : 該參數的名稱（例如 "port" 或 "num_threads"），用於錯誤訊息中讓
                        使用者看懂是哪個參數帶錯。    
*/
static long
getNum(const char *fname, const char *arg, int flags, const char *name)
{
    long res;
    char *endptr;
    int base;

    if (arg == NULL || *arg == '\0')
        gnFail(fname, "null or empty string", arg, name);

    /*
        • GN_ANY_BASE：設為 0。這代表 strtol 會自動偵測。如果字串以 0x 開頭就當 16 進位，
          以 0 開頭當 8 進位，否則當 10 進位。
        • GN_BASE_8：強制為 8 進位。
        • GN_BASE_16：強制為 16 進位。
        • 預設：若都沒設，則預設為 10 進位。
    */
    base = (flags & GN_ANY_BASE) ? 0 : (flags & GN_BASE_8) ? 8 :
                        (flags & GN_BASE_16) ? 16 : 10;

    /*
        • errno = 0 的必要性： strtol 在發生溢位（數值大於 LONG_MAX 或小於 LONG_MIN）
          時會設定全域變數 errno。因為 errno 不會自動歸零，所以在呼叫前必須手動清空。
        • endptr（終點指標）： strtol 成功轉換後，會把「第一個無法解析的字元位置」存入 endptr。
                            例如："123abc" → 轉換 123，endptr 指向 'a'。
        strtol() usage :
        long int strtol(const char *str, char **endptr, int base);
        str     要轉換的字串，可含空白、正負號、數字。
        endptr	若非 NULL，函式會把「第一個無法轉換的字元位置」寫入 *endptr。
        base	進位（2–36 或 0）。base=0 時會依字串前綴自動判斷進位。
                (base = 0 : 0x.0X -> 16進位; 0 -> 8進位; 其它 -> 10進位)
    */
    errno = 0;
    res = strtol(arg, &endptr, base);
    if (errno != 0)
        gnFail(fname, "strtol() failed", arg, name);

    if (*endptr != '\0')
        gnFail(fname, "nonnumeric characters", arg, name);

    if ((flags & GN_NONNEG) && res < 0)
        gnFail(fname, "negative value not allowed", arg, name);

    if ((flags & GN_GT_0) && res <= 0)
        gnFail(fname, "value must be > 0", arg, name);

    return res;
}

/* Convert a numeric command-line argument string to a long integer. See the
   comments for getNum() for a description of the arguments to this function. */

long
getLong(const char *arg, int flags, const char *name)
{
    return getNum("getLong", arg, flags, name);
}

/* Convert a numeric command-line argument string to an integer. See the
   comments for getNum() for a description of the arguments to this function. */

int
getInt(const char *arg, int flags, const char *name)
{
    long res;

    res = getNum("getInt", arg, flags, name);

    if (res > INT_MAX || res < INT_MIN)
        gnFail("getInt", "integer out of range", arg, name);

    return res;
}
