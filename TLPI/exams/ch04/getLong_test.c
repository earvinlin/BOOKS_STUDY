#include "../tlpi-book/mylib/tlpi_hdr.h"
#include "../tlpi-book/mylib/get_num.h"

int main(int argc, char *argv[]) {
    if (argc != 2)
        usageErr("%s <positive-integer>\n", argv[0]);

    long val = getLong(argv[1], GN_GT_0, "input-value");
    printf("Parsed value: %ld\n", val);
    
    return 0;
}
