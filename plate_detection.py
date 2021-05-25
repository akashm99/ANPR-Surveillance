import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
from object_detection.utils import visualization_utils as viz_utils
from utils.ocr import ocr_it
from utils.utils import save_results
from object_detection.utils import label_map_util
import tensorflow as tf


def gpu_config(memory=3120):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(gpus)
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)])
            print('success')
        except RuntimeError as e:
            print(e)

gpu_config()


class Predict:
    def __init__(self):
        PATH_TO_SAVED_MODEL = "training_op/models/saved_model/"
        PATH_TO_LABELS = "training_op/annotations/label_map.pbtxt"
        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
        self.detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)


    def predict_from_video(self, PATH_TO_VIDEO=0):

        # load frame and do inference
        cap = cv2.VideoCapture(PATH_TO_VIDEO)

        while cap.isOpened():
            ret, frame = cap.read()
            image_np = np.array(frame)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
            detections = self.detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 0
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + label_id_offset,
                detections['detection_scores'],
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.5,
                agnostic_mode=False)

            try:
                text, region= ocr_it(image=image_np_with_detections, detections=detections)
                save_results(text, region, 'detection_results.csv', 'detection_images')
                # print(text)
                # plt.show(region)
                # cv2.imshow('Number plate', cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            except:
                pass
            
            try:
                print(f"License Plate: {text}")
                cv2.imshow(f'{text}', cv2.resize(region, (100, 60)))
                #cv2.putText(image_np_with_detections, str(text), (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)
            except:
                pass

            cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

            if cv2.waitKey(0) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

    def predict_from_image(self, PATH_TO_IMAGE, api=False):
        img = cv2.imread(PATH_TO_IMAGE)
        image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
        detections = self.detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 0
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.5,
            agnostic_mode=False)

        try:
            text, region = ocr_it(image=image_np_with_detections, detections=detections)
            save_results(text, region, 'detection_results.csv', 'detection_images')
        except:
            pass
        # except Exception as error:
        #     print('Caught this error: ' + repr(error))

        if api == False:
            try:
                cv2.putText(image_np_with_detections, str(text), (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)
                print(f"License Plate: {text}")
                cv2.imshow('Number plate', cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            except:
                pass
            cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
            # to use it in a loop
            k = cv2.waitKey(0)
            if k == 27:  # wait for ESC key to exit
                cv2.destroyAllWindows()
        else:
            try:
                output = {"License_Plate": text,
                          "opImage": image_np_with_detections}
            except:
                output = "NO PLATE DETECTED, ENTER DIFFERENT IMAGE"

            return output


