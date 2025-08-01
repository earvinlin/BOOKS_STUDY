import zipfile
import os

def inspect_zip(zip_path):
    if not os.path.isfile(zip_path):
        print(f"❌ 檔案不存在：{zip_path}")
        return

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 檢查是否為有效的 ZIP 檔案
            if zip_ref.testzip() is not None:
                print("⚠️ ZIP 檔案可能損壞：", zip_ref.testzip())
            else:
                print("✅ ZIP 檔案結構正常。")
            
            # 列出所有檔案與資料夾
            print("\n📂 ZIP 檔案內容：")
            for info in zip_ref.infolist():
                print(f" - {info.filename} ({info.file_size} bytes)")
    except zipfile.BadZipFile:
        print("❌ 無法開啟 ZIP 檔案，可能已損壞或不是有效的 ZIP 格式。")
    except Exception as e:
        print(f"❌ 發生錯誤：{e}")

# 範例使用方式
zip_file_path = 'your_file.zip'  # 請替換為你的 ZIP 檔案路徑
inspect_zip(zip_file_path)
