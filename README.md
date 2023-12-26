# image2text

## 環境構築
```
pip install git+https://github.com/rionehome/image2text
```

## サンプルコード
```python
from image2text import i2t

for text in i2t(): # args: camera_id = 0, cache_dir="cache"
  print(text)

#  if (any_condition):
#    break
```

## 注意点
cacheのディレクトリを必ず.gitignoreしてください。