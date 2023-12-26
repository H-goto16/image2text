# image2text

## 環境構築
```
pip install git+https://github.com/rionehome/image2text
```

## サンプルコード
```python
import cv2
from image2text import i2t

i2t = i2t()

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# 1枚だけなら
text = i2t.image2text(frame)

# リアルタイムでしたい場合
for text in i2t.realtime_i2t():
    print(text)

```

## 注意点
cacheのディレクトリを必ず.gitignoreしてください。