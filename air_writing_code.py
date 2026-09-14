import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pytesseract
import os
import shutil
CAM_W = 640
CAM_H = 480
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except:
    pass
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0, min_detection_confidence=0.55, min_tracking_confidence=0.55)
paint = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
history = []
MAX_HISTORY = 30
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]
current_color = colors[0]
BRUSH_SIZE = 7
ERASER_SIZE = 35
prev_point = None
drawing = False
erasing = False
MAX_DRAW_JUMP = 90
last_color_change = 0
COLOR_COOLDOWN = 0.35
recognized_text = ''
ocr_message = ''
tesseract_paths = ['C:\\Program Files\\Tesseract-OCR\\tesseract.exe', 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe']
tesseract_found = False
for path in tesseract_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        tesseract_found = True
        break
if not tesseract_found and shutil.which('tesseract'):
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'
    tesseract_found = True
WINDOW_NAME = 'Air Writing Paint + AI'
DISPLAY_W = 900
DISPLAY_H = 600

def save_history():
    global history
    if len(history) > 0:
        if np.array_equal(history[-1], paint):
            return
    history.append(paint.copy())
    if len(history) > MAX_HISTORY:
        history.pop(0)

def undo():
    global paint
    global recognized_text
    global ocr_message
    if history:
        paint[:] = history.pop()
        recognized_text = ''
        ocr_message = 'UNDO DONE'
        print('Undo done')
    else:
        ocr_message = 'NOTHING TO UNDO'
        print('Nothing to undo')

def clear_canvas():
    global paint
    global recognized_text
    global ocr_message
    save_history()
    paint.fill(0)
    recognized_text = ''
    ocr_message = 'CANVAS CLEARED'
    print('Canvas cleared')

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def landmark_point(lm, index):
    return (int(lm[index].x * CAM_W), int(lm[index].y * CAM_H))

def finger_angle(a, b, c):
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return 0
    value = np.dot(ba, bc) / denominator
    value = np.clip(value, -1.0, 1.0)
    return math.degrees(math.acos(value))

def finger_is_up(lm, mcp, pip, dip, tip):
    angle1 = finger_angle(lm[mcp], lm[pip], lm[dip])
    angle2 = finger_angle(lm[pip], lm[dip], lm[tip])
    return angle1 > 145 and angle2 > 145

def get_fingers(lm):
    index = finger_is_up(lm, 5, 6, 7, 8)
    middle = finger_is_up(lm, 9, 10, 11, 12)
    ring = finger_is_up(lm, 13, 14, 15, 16)
    pinky = finger_is_up(lm, 17, 18, 19, 20)
    return (index, middle, ring, pinky)

def get_gesture(lm):
    index, middle, ring, pinky = get_fingers(lm)
    if index and (not middle) and (not ring) and (not pinky):
        return 'DRAW'
    if middle and (not index) and (not ring) and (not pinky):
        return 'COLOR'
    if index and middle and ring and pinky:
        return 'OPEN'
    if not index and (not middle) and (not ring) and (not pinky):
        return 'ERASE'
    return 'IDLE'

def select_color_from_x(x):
    global current_color
    start_x = 125
    for i in range(len(colors)):
        x1 = start_x + i * 55
        x2 = x1 + 40
        if x1 <= x <= x2:
            current_color = colors[i]
            print('Color selected:', i + 1)
            return

def draw_toolbar(frame):
    cv2.rectangle(frame, (10, 10), (630, 72), (25, 25, 25), -1)
    cv2.putText(frame, 'AIR PAINT', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    start_x = 125
    for i, col in enumerate(colors):
        x1 = start_x + i * 55
        y1 = 18
        x2 = x1 + 40
        y2 = 58
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1)
        if col == current_color:
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 2)
    cv2.putText(frame, 'I Write | M Color | Fist Erase', (470, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (230, 230, 230), 1)
    cv2.putText(frame, 'R Recognize | U Undo | C Clear', (470, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (190, 190, 190), 1)

def prepare_ocr_image():
    gray = cv2.cvtColor(paint, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    if cv2.countNonZero(mask) < 50:
        return None
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    padding = 30
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(CAM_W, x + w + padding)
    y2 = min(CAM_H, y + h + padding)
    crop = mask[y1:y2, x1:x2]
    crop = cv2.bitwise_not(crop)
    crop = cv2.copyMakeBorder(crop, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return crop

def recognize_air_writing():
    global recognized_text
    global ocr_message
    print('-----------------------------')
    print('AI Recognition started...')
    if not tesseract_found:
        recognized_text = ''
        ocr_message = 'TESSERACT NOT FOUND'
        print('Tesseract not found')
        return
    image = prepare_ocr_image()
    if image is None:
        recognized_text = ''
        ocr_message = 'WRITE SOMETHING FIRST'
        print('Canvas is empty')
        return
    candidates = []
    configs = ['--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ', '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ', '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    for config in configs:
        try:
            text = pytesseract.image_to_string(image, config=config)
            text = text.strip()
            text = ''.join((ch for ch in text.upper() if ch.isalpha() or ch == ' '))
            text = ' '.join(text.split())
            if text:
                candidates.append(text)
        except Exception as e:
            print('OCR method error:', e)
    gray = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 8 and h >= 15:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    char_result = ''
    for x, y, w, h in boxes:
        if h < 20:
            continue
        char_img = image[max(0, y - 10):min(image.shape[0], y + h + 10), max(0, x - 10):min(image.shape[1], x + w + 10)]
        try:
            char = pytesseract.image_to_string(char_img, config='--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ').strip()
            char = ''.join((ch for ch in char.upper() if ch.isalpha()))
            if char:
                char_result += char[0]
        except:
            pass
    if char_result:
        candidates.append(char_result)
    if candidates:
        candidates = list(dict.fromkeys(candidates))
        print('OCR candidates:', candidates)
        best = max(candidates, key=lambda x: len(x.replace(' ', '')))
        recognized_text = best
        ocr_message = 'RECOGNIZED'
        print('Final:', recognized_text)
    else:
        recognized_text = ''
        ocr_message = 'RECOGNITION FAILED'
        print('No result')
    print('-----------------------------')
print()
print('========================================')
print('       AIR WRITING PAINT + AI')
print('========================================')
print()
print('INDEX FINGER  = WRITE')
print('MIDDLE FINGER = COLOR')
print('CLOSED FIST   = ERASER')
print('OPEN HAND     = NOTHING')
print()
print('R = AI RECOGNIZE')
print('U = UNDO')
print('C = CLEAR')
print('S = SAVE')
print('Q = QUIT')
print()
if tesseract_found:
    print('Tesseract: READY')
else:
    print('Tesseract: NOT FOUND')
print()
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H)
while True:
    ret, frame = cap.read()
    if not ret:
        print('Camera frame nahi aa rahi.')
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True
    gesture = 'IDLE'
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3), mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2))
        index_point = landmark_point(lm, 8)
        cv2.circle(frame, index_point, 8, current_color, -1)
        cv2.circle(frame, index_point, 10, (255, 255, 255), 1)
        gesture = get_gesture(lm)
        if gesture == 'DRAW':
            if not drawing:
                save_history()
                prev_point = index_point
                drawing = True
            else:
                if prev_point is not None:
                    d = distance(prev_point, index_point)
                    if d <= MAX_DRAW_JUMP:
                        cv2.line(paint, prev_point, index_point, current_color, BRUSH_SIZE, cv2.LINE_AA)
                    else:
                        prev_point = index_point
                prev_point = index_point
            erasing = False
        elif gesture == 'COLOR':
            drawing = False
            erasing = False
            prev_point = None
            now = time.time()
            if now - last_color_change > COLOR_COOLDOWN:
                middle_point = landmark_point(lm, 12)
                cv2.circle(frame, middle_point, 12, (255, 255, 255), 2)
                select_color_from_x(middle_point[0])
                last_color_change = now
        elif gesture == 'ERASE':
            if not erasing:
                save_history()
                erasing = True
                drawing = False
                prev_point = None
            palm_x = int((lm[0].x + lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 5 * CAM_W)
            palm_y = int((lm[0].y + lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 5 * CAM_H)
            eraser_point = (palm_x, palm_y)
            cv2.circle(frame, eraser_point, ERASER_SIZE, (255, 255, 255), 2)
            cv2.circle(paint, eraser_point, ERASER_SIZE, (0, 0, 0), -1, cv2.LINE_AA)
        else:
            drawing = False
            erasing = False
            prev_point = None
    else:
        drawing = False
        erasing = False
        prev_point = None
    output = cv2.add(frame, paint)
    draw_toolbar(output)
    if gesture == 'DRAW':
        mode_color = current_color
    elif gesture == 'ERASE':
        mode_color = (255, 255, 255)
    elif gesture == 'COLOR':
        mode_color = (0, 255, 255)
    else:
        mode_color = (180, 180, 180)
    cv2.rectangle(output, (15, 90), (155, 125), (20, 20, 20), -1)
    cv2.putText(output, gesture, (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)
    cv2.rectangle(output, (15, 405), (625, 465), (20, 20, 20), -1)
    cv2.putText(output, 'AI TEXT:', (25, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    display_text = recognized_text
    if len(display_text) > 32:
        display_text = display_text[:32]
    cv2.putText(output, display_text, (120, 432), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(output, ocr_message, (25, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
    cv2.imshow(WINDOW_NAME, output)
    key = cv2.waitKey(1) & 255
    if key == ord('r') or key == ord('R'):
        recognize_air_writing()
    elif key == ord('u') or key == ord('U'):
        undo()
    elif key == ord('c') or key == ord('C'):
        clear_canvas()
    elif key == ord('s') or key == ord('S'):
        cv2.imwrite('air_writing.png', paint)
        ocr_message = 'SAVED'
        print('Saved as air_writing.png')
    elif key == ord('q') or key == 27:
        break
cap.release()
hands.close()
cv2.destroyAllWindows()
print()
print('Air Writing Paint closed.')
