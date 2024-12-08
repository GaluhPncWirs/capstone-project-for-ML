from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def create_image_toText(text, img_width=32, img_height=32, font_size=20, font_path="font/Helvetica.ttf"):
    # Membuat gambar kosong berwarna putih
    img = Image.new('L', (img_width, img_height), color=255)
    draw_img = ImageDraw.Draw(img)
    
    # Memuat font
    font = ImageFont.truetype(font_path, font_size)

    margin = 5
    
    # Menghitung bounding box teks
    text_bbox = draw_img.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Menghitung posisi agar teks berada di tengah
    x = (img_width - text_width) // 2
    y = (img_height - text_height) // 2


    x = max(0, x - margin)
    y = max(0, y - margin)
    
    # Menggambar teks di posisi tengah
    draw_img.text((x, y), text, font=font, fill=0)
    return np.array(img)

# Daftar huruf
# text_to_img = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# text_to_img = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# text_to_img = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
text_to_img = ['.',',', '%', ':', ';', '<', '>', '/', '+', '=', '-', '(', ')', '*', '[', ']', '`', '!']


# Folder output
output_folder = "symbol"
os.makedirs(output_folder, exist_ok=True)

# Proses pembuatan gambar
image_paths = []
for i, text in enumerate(text_to_img):
    try:
        img_array = create_image_toText(text)
        img = Image.fromarray(img_array)
        image_path = os.path.join(output_folder, f"symbol{i}.png")
        img.save(image_path)
        image_paths.append(image_path)
    except Exception as e:
        print(f"Error processing text '{text}': {e}")

print("Proses selesai! Data gambar dan label berhasil disimpan.")