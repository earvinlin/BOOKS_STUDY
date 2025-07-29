/*
 * CH16 問答題
 */

// 01 (a)
// 在 C 語言中，全域變數的初始化值必須是常數，不能是運算結果或函式返回值。
#define FPM 5280
int miles = 5;

int dist = FPM * 2;
//dist = FPM;

/***
// 01 (b)
#define FEET 4
#define POD FEET + FEET
plort = FEET * POD

// 01 (c)
#define SIX = 6l
nex = SIX;

// 01 (d)
#define NEW(X) X + %
y = NEW(y);
berg = NEW(berg) * lob;
est = NEW() / NEW(y);
nilp = lob * NEW(-berg);
*/

int main()
{
	float dist2 = FPM * miles;

	printf("%i\n", dist);
	printf("%f\n", dist2);

	return 0;
}