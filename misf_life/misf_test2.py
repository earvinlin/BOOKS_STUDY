import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import csv
results = []

# ====== 使用者帳密 ======
ACCOUNT = "AAA"
PASSWORD = "BBB"

# ====== 定義：執行一次 Tab 切換測試 ======
def run_tab_test(round_no: int):
    log_time(f"===== 開始第 {round_no} 次 Tab 切換測試 =====")

    tabs = driver.find_elements(By.CSS_SELECTOR, ".ant-tabs-tab")

    for idx, tab in enumerate(tabs):
        try:
            log_time(f"準備點擊 Tab {idx+1}")

            start_tab = time.time()

            tab.click()
            time.sleep(1)

            end_tab = time.time()

            elapsed = (end_tab - start_tab) * 1000
            log_time(f"Tab {idx+1} 切換完成，耗時 {elapsed:.2f} ms")

            # ★★★ 正確：每個 tab 都要記錄 ★★★
            results.append({
                "round": round_no,
                "tab": idx + 1,
                "elapsed_ms": f"{elapsed:.2f}"
            })

        except Exception as e:
            print(f"Tab {idx+1} 無法點擊：{e}")

    log_time(f"===== 第 {round_no} 次 Tab 切換測試完成 =====\n")

# ====== 記錄時間的工具函式 ======
def log_time(event: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] {event}")

# ====== 啟動 Chrome ======
options = Options()
options.add_experimental_option("detach", True)

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

wait = WebDriverWait(driver, 10)

# ====== 開啟登入頁 ======
start = time.time()
driver.get("https://elife.twfhclife.com.tw/MISF/mis/login")
wait.until(EC.presence_of_element_located((By.NAME, "memberId")))
log_time("登入頁載入完成")

# ====== 找帳號欄位 ======
account_input = wait.until(
    EC.presence_of_element_located((By.NAME, "memberId"))
)
password_input = wait.until(
    EC.presence_of_element_located((By.NAME, "passPhrase"))
)
log_time("帳密欄位載入完成")

# ====== 自動輸入帳密 ======
account_input.send_keys(ACCOUNT)
password_input.send_keys(PASSWORD)
log_time("帳密輸入完成")

# 等待10秒手動輸入圖形識別數字及OTP驗證碼
time.sleep(10)

# ====== 按登入 ======
login_button = wait.until(
    EC.element_to_be_clickable((By.ID, "loginBtn"))
)
# login_button.click()
log_time("登入按鈕已點擊")

# ====== 等待登入後頁面載入 ======
# 依照實際登入後頁面調整等待條件
wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
log_time("登入後頁面載入完成")


"""
# 點擊所有 Tabs（未提交 / 未簽署 / 進件中…）
tabs = driver.find_elements(By.CSS_SELECTOR, ".ant-tabs-tab")

for idx, tab in enumerate(tabs):
    try:
        print(f"點擊 Tab {idx+1}")
        tab.click()
        time.sleep(1)
    except Exception as e:
        print(f"Tab {idx+1} 無法點擊：{e}")


# ====== 點擊所有 Tabs（未提交 / 未簽署 / 進件中…） ======
tabs = driver.find_elements(By.CSS_SELECTOR, ".ant-tabs-tab")

for idx, tab in enumerate(tabs):
    try:
        log_time(f"準備點擊 Tab {idx+1}")

        start_tab = time.time()  # 開始計時

        tab.click()
        time.sleep(1)  # 等待頁面切換（可依需求調整）

        end_tab = time.time()  # 結束計時

        elapsed = (end_tab - start_tab) * 1000  # 轉毫秒
        log_time(f"Tab {idx+1} 切換完成，耗時 {elapsed:.2f} ms")

    except Exception as e:
        print(f"Tab {idx+1} 無法點擊：{e}")
"""

# ====== 連續執行 10 次 Tab 切換 ======
for i in range(1, 11):
    run_tab_test(i)
    time.sleep(5)  # 每輪之間稍微休息一下（可調整）

csv_filename = "tab_performance.csv"
with open(csv_filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["round", "tab", "elapsed_ms"])
    writer.writeheader()
    writer.writerows(results)

log_time(f"CSV 已輸出：{csv_filename}")