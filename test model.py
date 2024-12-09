import cv2
import re
import json
import logging
from paddleocr import PaddleOCR
import pytesseract
import numpy as np

# # Inisialisasi PaddleOCR
# ocr = PaddleOCR(
#     use_angle_cls=True,
#     lang='en'  # Menggunakan model bahasa Inggris
# )

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Mengonversi gambar menjadi grayscale
def convert_2_gray(image):
    if image is None:
        raise ValueError("Gambar tidak ditemukan atau tidak valid.")
    if len(image.shape) == 3:  # Hanya konversi jika gambar memiliki 3 channel
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# Binarisasi gambar
def binarization(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

# Menemukan blok teks
def find_text_blocks(image):
    binary_image = binarization(image)  # Binarisasi gambar
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 20:  # Hanya blok teks yang cukup besar
            blocks.append([x, y, w, h])

    # Urutkan blok teks berdasarkan posisi top-to-bottom, left-to-right
    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
    return blocks

## Koreksi rotasi gambar
def deskew_image(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return deskewed

# Preprocessing untuk OCR
def preprocessing_gambar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Tingkatkan kontras
    contrast = cv2.equalizeHist(gray)
    
    # Hilangkan noise
    noise_removed = cv2.medianBlur(contrast, 3)
    
    # Binarisasi
    _, binary = cv2.threshold(noise_removed, 150, 255, cv2.THRESH_BINARY)
    
    # Perbesar gambar
    # enlarged = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return binary

# Ekstraksi teks dari blok
def extract_text_from_block(image):
    processed_image = preprocessing_gambar(image)
    text_blocks = find_text_blocks(processed_image)
    
    if not text_blocks:
        return "No text found."

    full_text = ""
    for block in text_blocks:
        x, y, w, h = block
        block_img = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(block_img, config='--oem 3 --psm 6', lang='ind')  # Mode untuk blok teks
        full_text += text.strip() + "\n"

    return full_text.strip()

# Membaca teks dari gambar
def read_story_text(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Tidak dapat membaca gambar dari path yang diberikan.")
    text = extract_text_from_block(image)
    return text


def parse_nutrition_info(text):
    nutrition_data = {}

    # semua kemungkinan yang ada di informasi nilai gizi
    keywords = {
        'Takaran Saji': 'takaran saji',
        'Sajian per Kemasan': 'sajian per kemasan',
        'Energi Total': 'energi total',
        'Energi dari Lemak': 'energi dari lemak',
        'Lemak Total': 'lemak total',
        'Lemak Jenuh': 'lemak jenuh',
        'Lemak Trans': 'lemak trans',
        'Kolesterol': 'kolesterol',
        'Protein': 'protein',
        'Karbohidrat Total': 'karbohidrat total',
        'Serat Pangan': 'serat pangan',
        'Gula': 'gula',
        'Gula Total': 'gula total',
        'Sugars': 'sugars',
        'Garam': 'garam',
        'Sodium': 'sodium',
        'Kalium': 'kalium',
        'Potassium': 'potassium',
    }

    # Prioritas kata kunci untuk menghindari duplikasi
    priority_keywords = {
        "gula total": "gula",  # Prioritaskan "gula"
    }

    # Perbaikan teks untuk menangani kesalahan umum OCR
    text = text.replace('9 g', '9g')
    text = re.sub(r'(\d)9\b', r'\1g', text)  # Ubah '9' menjadi 'g' jika di akhir kata
    text = re.sub(r'\b9g\b', '9 g', text)      # Tangani kasus '9g' menjadi hanya '9 g'
    text = re.sub(r'(\d)(\s?)(g)', r'\1 g', text)  # Menambahkan spasi antara angka dan unit 'g'

    # Default satuan untuk setiap jenis nutrisi
    default_units = {
        "takaran saji": "g",
        "takaran saji": "ml",
        "energi total": "kkal",
        "energi dari lemak": "kkal",
        "lemak total": "g",
        "lemak jenuh": "g",
        "protein": "g",
        "karbohidrat total": "g",
        "gula": "g",
        "garam": "mg",
        "kolesterol": "mg",
        "serat pangan": "g",
        "kalium": "mg",
    }

    # Proses parsing teks
    lines = text.splitlines()
    for line in lines:
        for keyword, nutrient in keywords.items():
            if keyword.lower() in line.lower():
                # Cari angka dan satuan
                match = re.search(r'(\d+(?:[.,]\d+)?)\s*(g|mg|kg|kkal|ml)?', line.lower())
                if match:
                    value = match.group(1)  # Ambil angka
                    unit = match.group(2) or default_units.get(nutrient, "")  # Gunakan satuan default jika tidak ada
                    nutrition_data[nutrient] = f"{value} {unit}".strip()

    # Hapus item dengan nilai kosong atau dimulai dengan '0'
    to_delete = [key for key, value in nutrition_data.items() if value.startswith("0") or value == ""]
    for key in to_delete:
        del nutrition_data[key]

    # Hapus duplikasi berdasarkan prioritas
    for lower_priority, higher_priority in priority_keywords.items():
        if lower_priority in nutrition_data and higher_priority in nutrition_data:
            del nutrition_data[lower_priority]  # Hapus item dengan prioritas lebih rendah

    # Pastikan setiap nutrisi memiliki satuan default jika tidak terdeteksi
    for nutrient, default_unit in default_units.items():
        if nutrient in nutrition_data and not any(nutrition_data[nutrient].endswith(u) for u in ["g", "mg", "kkal", "ml"]):
            nutrition_data[nutrient] += f" {default_unit}"

    return nutrition_data

# Test the process
image_path = './testing/images/ing-8.jpg'  # Replace with your image path

try:
    # Read the text from the image
    story_text = read_story_text(image_path)
    print("Extracted Text from Image:\n")
    print(story_text)  # Debugging: print the OCR result

    # Parse the nutrition information from the extracted text
    nutrition_info = parse_nutrition_info(story_text)
    
    # Convert the parsed data to JSON format with indentation
    nutrition_json = json.dumps(nutrition_info, indent=4)
    
    print("\nParsed Nutrition Information in JSON Format:\n")
    print(nutrition_json)
    
except Exception as e:
    print(f"Error occurred while reading the image or processing the text: {e}")