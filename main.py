import cv2
import numpy as np
from sort.sort import Sort
from datetime import datetime
import os

# Archivos del modelo
cfg_file = "darknet/cfg/yolov3.cfg"
weights_file = "darknet/cfg/yolov3.weights"
names_file = "darknet/data/coco.names"

if not os.path.isfile(cfg_file):
    raise FileNotFoundError(f"El archivo de configuración {cfg_file} no se encuentra en el directorio.")
if not os.path.isfile(weights_file):
    raise FileNotFoundError(f"El archivo de pesos {weights_file} no se encuentra en el directorio.")
if not os.path.isfile(names_file):
    raise FileNotFoundError(f"El archivo de nombres {names_file} no se encuentra en el directorio.")

# Inicializar el modelo de detección
net = cv2.dnn.readNet(weights_file, cfg_file)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Inicializar el tracker SORT
tracker = Sort(iou_threshold=0.5)

# Diccionario para almacenar tiempos de entrada y salida
entry_exit_times = {}

# Diccionario para almacenar el estado de cada ID
person_state = {}

# Inicializar la cámara (usa 0 para la cámara predeterminada)
cap = cv2.VideoCapture("src/video.mp4")

# Establecer el umbral de confianza para YOLO
confidence_threshold = 0.7  # Ajusta este valor según sea necesario

# Función para calcular el IOU
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    return intersection / union

# Función para filtrar detecciones duplicadas
def filter_duplicate_detections(detections, iou_threshold=0.5):
    if len(detections) == 0:
        return detections

    filtered_detections = []
    for i in range(len(detections)):
        keep = True
        for j in range(i + 1, len(detections)):
            iou = calculate_iou(detections[i], detections[j])
            if iou > iou_threshold:
                keep = False
                break
        if keep:
            filtered_detections.append(detections[i])
    return np.array(filtered_detections)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Recopilar detecciones de personas
    detections = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > confidence_threshold:  # Clase '0' es 'persona'
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                detections.append([x, y, x + w, y + h, confidence])

    # Filtrar detecciones duplicadas
    detections = filter_duplicate_detections(detections)

    # Asegurarse de que detections tenga la forma correcta
    if len(detections) > 0:
        detections = np.array(detections)
        if detections.ndim == 1:
            detections = np.expand_dims(detections, axis=0)
    else:
        detections = np.array([]).reshape(0, 5)

    # Actualizar el tracker con las detecciones actuales
    trackers = tracker.update(detections)

    # Dibujar los cuadros delimitadores y registrar entrada/salida
    current_time = datetime.now()
    for trk in trackers:
        x1, y1, x2, y2, trk_id = trk
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        if trk_id not in entry_exit_times:
            # Nueva persona detectada
            entry_exit_times[trk_id] = {'entry': current_time, 'exit': None}
            person_state[trk_id] = 'entered'
        else:
            # Verificar si ya se contó recientemente
            if person_state[trk_id] == 'entered':
                entry_exit_times[trk_id]['exit'] = current_time
                person_state[trk_id] = 'exited'

    # Mostrar el cuadro
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()

# Imprimir los tiempos de entrada y salida
for trk_id, times in entry_exit_times.items():
    print(f"Persona {trk_id}: Entrada - {times['entry']}, Salida - {times['exit']}")