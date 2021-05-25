import numpy as np
import easyocr

def filter_text(region, ocr_result, region_threshold=0.3):
    rectangle_size = region.shape[0] * region.shape[1]
    plate = []

    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        if length * height / rectangle_size > region_threshold:
            plate.append(result[1])

    return plate

def ocr_it(image, detections, detection_threshold=0.6, region_threshold=0.3):
    scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]

    width = image.shape[1]
    height = image.shape[0]

    if len(boxes) != 0:
        for box in boxes:
            roi = box * [height, width, height, width]
            region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]
            reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            ocr_result = reader.readtext(region)
            text = filter_text(region, ocr_result, region_threshold)
            # plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            return text, region

