from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

# ====== 使用者帳密 ======
ACCOUNT = "A121539472"
PASSWORD = "E98321600l@"

# ====== 啟動 Chrome ======
options = Options()
options.add_experimental_option("detach", True)  # 保持瀏覽器不關閉

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

driver.get("https://elife.twfhclife.com.tw/MISF/mis/login")

# ====== 等待欄位載入 ======
wait = WebDriverWait(driver, 10)

# 找帳號欄位
account_input = wait.until(
    EC.presence_of_element_located((By.NAME, "memberId"))
)

# 找密碼欄位
password_input = wait.until(
    EC.presence_of_element_located((By.NAME, "passPhrase"))
)

# ====== 自動輸入帳密 ======
account_input.send_keys(ACCOUNT)
password_input.send_keys(PASSWORD)

# ====== 自動按登入（可選） ======

time.sleep(10)


login_button = wait.until(
    EC.element_to_be_clickable((By.ID, "loginBtn"))
)
login_button.click()

