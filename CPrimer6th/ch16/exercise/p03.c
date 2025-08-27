#include <stdio.h>
#include <math.h>

#define RAD_TO_DEG (180.0 / M_PI)  // 將弧度轉換為角度

typedef struct {
    double x;
    double y;
} RectCoord;

typedef struct {
    double r;       // 徑向距離
    double theta;   // 角度（度數）
} PolarCoord;

// 轉換函數：直角座標 → 極座標
PolarCoord rect_to_polar(RectCoord rc) {
    PolarCoord pc;
    pc.r = sqrt(rc.x * rc.x + rc.y * rc.y);         // 計算距離
    pc.theta = atan2(rc.y, rc.x) * RAD_TO_DEG;      // 計算角度（度）
    return pc;
}

int main() {
    RectCoord input;
    PolarCoord result;

    printf("請輸入 x 和 y 座標（例如：3 4）：\n");
    while (scanf("%lf %lf", &input.x, &input.y) == 2) {
        result = rect_to_polar(input);
        printf("極座標：r = %.2f, θ = %.2f 度\n", result.r, result.theta);
        printf("請再輸入 x 和 y 座標（或 Ctrl+D 結束）：\n");
    }

    printf("程式結束，再見！\n");
    return 0;
}
