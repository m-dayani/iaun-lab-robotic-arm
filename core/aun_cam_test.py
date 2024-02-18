import cv2
import time
import os


def capture_video():
    captured_duration=200
    cap = cv2.VideoCapture(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('U', 'Y', 'V', 'Y'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Width=", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height=", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        print("Error reading video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = os.getenv('DATA_PATH')
    if out_dir is None:
        out_dir = '.'

    result = cv2.VideoWriter(os.path.join(out_dir, 'output.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30, (640, 480))
    image_dir = os.path.join('.', out_dir, 'image')
    img_ext = '.png'
    i = 0

    start_time = time.time()

    while int(time.time() - start_time) < captured_duration:

        ret, frame = cap.read()
        if ret:
            # cv2.imwrite(image_dir + str(i) + img_ext, frame)
            i += 1
            result.write(frame)
            cv2.imshow("OpenCVCam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    result.release()
    cv2.destroyAllWindows()


def capture_image(i):

    cam_port = -1
    cam = cv2.VideoCapture(cam_port)

    result, image = cam.read()

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Width=", cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height=", cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = os.getenv('DATA_PATH')
    if out_dir is None:
        out_dir = '.'

    if result:
        cv2.imshow("Image", image)
        cv2.waitKey(0)

        # out_dir = 'data'
        # image_dir = os.path.join('.', out_dir, 'image')
        img_ext = '.png'

        image_file = os.path.join(out_dir, 'image' + str(i) + img_ext)

        cv2.imwrite(image_file, image)
        print("image is written: " + image_file)

        # cv2.destroyAllWindows()
    else:
        print("No image detected")

    cam.release()


if __name__ == '__main__':

    capture_video()

    # for i in range(10):
    #     capture_image(50+i)
