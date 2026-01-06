import requests
import os
import time

BASE_URL = "https://laws.e-gov.go.jp/api/2/law_data/" # Endpoint chính xác cho v2
OUTPUT_DIR = "data/xml_raw"
INPUT_FILE = "law_ids.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_fix():
    with open(INPUT_FILE, "r") as f:
        law_ids = [line.strip() for line in f if line.strip()]

    for idx, law_id in enumerate(law_ids):
        # Construct full URL for v2: BASE_URL + law_id
        url = f"{BASE_URL}{law_id}"
        headers = {"Accept": "application/xml"}
        
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                file_path = os.path.join(OUTPUT_DIR, f"{law_id}.xml")
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ [{idx+1}/{len(law_ids)}] Đã tải: {law_id}")
            else:
                print(f"❌ [{idx+1}/{len(law_ids)}] Lỗi {law_id}: Status {response.status_code}")
                # Log thử nội dung lỗi nếu có
                # print(response.text) 
                
            time.sleep(1.2) # Tránh bị rate limit
        except Exception as e:
            print(f"⚠️ Lỗi kết nối {law_id}: {e}")

download_fix()