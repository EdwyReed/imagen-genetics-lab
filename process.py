# pip install pillow
from pathlib import Path
from PIL import Image, ImageOps

SRC = Path("alpha")
DST = Path("alpha_processed")
DST.mkdir(parents=True, exist_ok=True)

# Соберём файлы и отсортируем по имени
files = sorted([p for p in SRC.iterdir() if p.is_file()])

i = 1
for p in files:
    try:
        with Image.open(p) as im:
            # Учитываем EXIF-ориентацию
            im = ImageOps.exif_transpose(im)

            w, h = im.size
            if w == 0 or h == 0:
                continue

            # Масштабируем так, чтобы большая сторона стала 512
            scale = 512 / max(w, h)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            im = im.resize(new_size, Image.LANCZOS)

            # Для JPEG нужен RGB; если RGBA — кладём на белый фон
            if im.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
            elif im.mode != "RGB":
                im = im.convert("RGB")

            out_path = DST / f"{i}.jpg"
            im.save(out_path, format="JPEG", quality=92, optimize=True)
            i += 1
    except Exception:
        # Пропускаем файлы, которые не удалось открыть как изображение
        continue
