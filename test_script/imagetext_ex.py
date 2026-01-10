from PIL import Image
from imagetext_py import *
import os

# 폰트 파일을 직접 로드
# LoadFromPath(name, path) 형태: name은 FontDB에 등록할 이름, path는 파일 경로
font_path = os.path.abspath(os.path.join("fonts", "NotoEmoji-Regular.ttf"))
FontDB.LoadFromPath("NotoEmoji", font_path)

# LoadFromPath에서 사용한 name과 동일한 이름으로 쿼리
font = FontDB.Query("NotoEmoji")

# create a canvas to draw on
cv = Canvas(512, 512, (255, 255, 255, 255))

# paints are used to fill and stroke text
black = Paint.Color((0, 0, 0, 255))
rainbow = Paint.Rainbow((0.0,0.0), (256.0,256.0))

# if a font doesn't have a glyph for a character, it will use the fallbacks
text = "hello my 😓 n🐢ame i☕s 会のすべ aての構成員 nathan and i drink soup boop coop, the quick brown fox jumps over the lazy dog"

draw_text_wrapped(canvas=cv,              # the canvas to draw on
                  text=text,
                  x=256, y=256,           # the position of the text
                  ax=0.5, ay=0.5,         # the anchor of the text
                  size=67,                # the size of the text
                  width=500,              # the width of the text
                  font=font,
                  fill=black,
                  align=TextAlign.Center,
                  stroke=2.0,             # the stroke width (optional)
                  stroke_color=rainbow,
                  draw_emojis=True)   # the stroke color (optional)

# you can convert the canvas to a PIL image
im: Image.Image = cv.to_image()
im.save("test.png")

# or you can just get the raw bytes
dimensions, bytes = cv.to_bytes()

# you can also save directly to a file
cv.save("test.png")


## USE PIL

with Image.new("RGBA", (512, 512), "white") as im:
    with Writer(im) as w:
        w.draw_text_wrapped(
            text = "hello my 😓 n🐢ame i☕s 会のすべ ",
            x=256, y=256,
            ax=0.5, ay=0.5,
            width=500,
            size=90,
            font=font,
            fill=Paint.Color((0, 0, 0, 255)),
            align=TextAlign.Center,
            stroke=2.0,
            stroke_color=Paint.Rainbow((0.0,0.0), (256.0,256.0)),
            draw_emojis=True
        )
    im.save("testPIL.png")