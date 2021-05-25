from plate_detection import Predict

PATH_TO_SAVED_MODEL = "training_op/models/saved_model/"
PATH_TO_LABELS = "training_op/annotations/label_map.pbtxt"
predict = Predict(PATH_TO_LABELS, PATH_TO_SAVED_MODEL)

#if predict from Video
PATH_TO_VIDEO = "videoplayback.mp4"
predict.predict_from_video(PATH_TO_VIDEO)


#if predict from Image
PATH_TO_IMAGE = "training_op/test_images/Cars401.png"
predict.predict_from_image(PATH_TO_IMAGE)

