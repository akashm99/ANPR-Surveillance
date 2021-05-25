import cv2, os, csv, uuid

def save_results(text, region, csv_filename, folder_path):
    img_name = f'{uuid.uuid1()}.jpg'
    cv2.imwrite(os.path.join(folder_path, img_name), region)

    with open(csv_filename, mode="a", newline='') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow([img_name, text])