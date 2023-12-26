import cv2
from i2t_from_frame import i2t_from_frame

def i2t(camera_id=0, cache_dir="cache"):
    """
    + image2text
    ```python
    from image2text import i2t

    for text in i2t():
        print(text)
    ```

    args:
        camera_id: camera id, default: 0
        cache_dir: cache directory, default: "cache"
    yield:
        text: text of image
    """
    cap = cv2.VideoCapture(camera_id)
    while True:
        ret, frame = cap.read()
        if not ret:
            pass
        i2t_from_frame(frame, cache_dir=cache_dir)
        cv2.imshow("image2text", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()