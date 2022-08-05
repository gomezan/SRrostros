
import cv2


#Daddas los atributos de entrada se crea un string que contiene la configuración del pipeline de la libreria gstreamer
def ajustePipeline(
    sensor_id=0,
    capture_width=512,
    capture_height=512,
    display_width=1080,
    display_height=720,
    framerate=100,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
