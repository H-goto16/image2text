from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

def i2t_from_frame(frame, cache_dir="cache"):
    """
    + image2text
    ```python
    from image2text import i2t_from_frame

    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        text = i2t_from_frame(frame)
        print(text)
    ```
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir=cache_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device: ", device.upper())
    if (device == "cuda"):
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16, cache_dir=cache_dir).to("cuda")
    else:
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir=cache_dir)

    while True:
        if device == "cuda":
            processor=processor(frame, "a photography of", return_tensors="pt").to(device, torch.float16)
        else:
            inputs = processor(frame, "a photography of", return_tensors="pt")
        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)
