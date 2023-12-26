import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class i2t:
  def __init__(self, cache_dir="cache", camera_id=0, using_gpu=False):
    self.camera_id = camera_id
    self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir=cache_dir)
    self.device = "cuda" if torch.cuda.is_available() or using_gpu else "cpu"

    print("Using device: ", self.device.upper())
    if (self.device == "cuda"):
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16, cache_dir=cache_dir).to("cuda")
    else:
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir=cache_dir)

  def image2text(self, frame):
    if self.device == "cuda":
        inputs=self.processor(frame, "a photography of", return_tensors="pt").to(self.device, torch.float16)
    else:
        inputs = self.processor(frame, "a photography of", return_tensors="pt")
    out = self.model.generate(**inputs)
    return self.processor.decode(out[0], skip_special_tokens=True)

  def realtime_i2t(self):
    """
    + image2text
    ```python
    from image2text import i2t
    i2t = i2t()
    for text in i2t.realtime_i2t():
        print(text)
    ```
    args:
        camera_id: camera id, default: 0
    yield:
        text: text of image
    """
    cap = cv2.VideoCapture(self.camera_id)
    while True:
        ret, frame = cap.read()
        if not ret:
            pass
        text = self.image2text(frame)
        cv2.imshow("image2text", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        yield text
    cap.release()