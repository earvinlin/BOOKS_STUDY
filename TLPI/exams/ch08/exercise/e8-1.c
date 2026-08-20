/**
 * 執行下列程式碼時，會顯示兩個不同使用者ID 的使用者名稱，我們發現程式將相同的使用者名稱顯示兩次。請問為什麼？
 * printf (%s %s", getwuid(uid1)->pw_name,
 * getwuid (uid2) ->pw_name);
 */
#include <stdlib.h>
#include <errno.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

int main(int argc, char *argv[])
{
    
    return 0;
}