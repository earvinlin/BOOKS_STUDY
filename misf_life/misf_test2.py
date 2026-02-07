import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# ====== 使用者帳密 ======
ACCOUNT = "AAA"
PASSWORD = "BBB"

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