import numpy as np
import cv2
from rknnlite.api import RKNNLite
from PIL import Image, ImageDraw, ImageFont
import time
import eventlet
import socketio
import base64
import os
import time
from flask import Flask, request, jsonify


sio = socketio.Server(cors_allowed_origins="*")
app = socketio.WSGIApp(sio)
script_dir = os.path.dirname(os.path.realpath(__file__))
# flask_app = Flask(__name__)


plate_chr = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"


OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = 640

CLASSES = ("plate", )

def decodePlate(preds):  # 识别后处理
    pre = 0
    newPreds = []
    for i in range(len(preds[0])):
        if (preds[0][i] != 0).all() and (preds[0][i] != pre).all():
            newPreds.append(preds[0][i])
        pre = preds[0][i]
    plate = ""
    for i in newPreds:
        plate += plate_chr[int(i)]
    return plate


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def sigmoid(x):
    # return 1 / (1 + np.exp(-x))
    return x

def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    box_wh = pow(sigmoid(input[..., 2:4])*2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score* box_confidences)[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


# environ
@sio.event
def connect(sid, environ):
    print('connect ', sid)

@sio.event
def my_message(sid, data):
    print('sending message ', data)
    sio.emit('my_message', 'Server received your message', room=sid)

@sio.event
def screenshot(sid, data):
    
    img_decoded = base64.b64decode(data)
    img_array = np.frombuffer(img_decoded, np.uint8)
    print("Received image data length:", len(img_array))
    # 解码图像
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        print("Error: Decoded image is None")
        img_back = base64.b16encode(img_decoded).decode('utf-8')
        sio.emit('processed', img_back, sid)
    else:
        print("Image decoded successfully")
        
        fps = 15 + np.random.rand() # 模拟帧率
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Inference
        # print('--> Running model')
        outputs = rknn1.inference(inputs=[img])
        # print('done')

        # post process
        input0_data = outputs[0]
        input1_data = outputs[1]
        input2_data = outputs[2]

        input0_data = input0_data.reshape([3, -1] + list(input0_data.shape[-2:]))
        input1_data = input1_data.reshape([3, -1] + list(input1_data.shape[-2:]))
        input2_data = input2_data.reshape([3, -1] + list(input2_data.shape[-2:]))

        input_data = list()
        input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

        boxes, classes, scores = yolov5_post_process(input_data)

        img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if boxes is not None:
            # draw(img_1, boxes, scores, classes)
            for box in boxes:
                x1, y1, x2, y2 = box
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                cv2.rectangle(img_1, (x1, y1), (x2, y2), (0, 255, 0), 1, 1)
                roi = img_1[y1:y2, x1:x2]
                try:
                    roi = cv2.resize(roi, (168, 48))
                    output = rknn2.inference(inputs=[roi])
                    input_data = np.swapaxes(output[0], 1, 2)
                    index = np.argmax(input_data, axis=1)
                    plate_no = decodePlate(index)
                    img_1 = cv2ImgAddText(img_1, str(plate_no), x1, y1 - 30, (0, 255, 0), 30)
                    # print(plate_no)
                except:
                    continue
        else :
            str_no = "未检测到车牌"
            img_1 = cv2ImgAddText(img_1, str_no, 300, 10, (0, 255, 0), 30)

        img = img_1
        cv2.putText(img, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2)  # 在图像上显示帧率

        # 编码处理后的图像
        img_back = base64.b64encode(cv2.imencode('.png', img)[1])
        print("Image processed and encoded successfully")
        print("Sending processed image data length:", len(img_back))
        # 发送处理后的图像数据回客户端
        sio.emit('processed', img_back, sid)

        # 释放OpenCV资源
        cv2.waitKey(1)  # 添加小延迟以便让OpenCV刷新窗口
        

@sio.event
def disconnect(sid):
    print('disconnect ', sid)


if __name__ == '__main__':

    rknn1 = RKNNLite(verbose=False)
    rknn2 = RKNNLite(verbose=False)
    rknn1.load_rknn("./yolov5n.rknn")
    rknn2.load_rknn("./best.rknn")

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn1.init_runtime()
    ret = rknn2.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')
    
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)

    """
    # Set inputs
    cap = cv2.VideoCapture('./test1.mp4')  #video视频
    frames, loopTime, initTime = 0, time.time(), time.time()
    fps = 0
    cnt_plate = 0
    while True:
        frames += 1
        # 从摄像头捕获帧
        ret, img = cap.read()
        # 如果捕获到帧，则显示它
        if ret:
            if frames % 30 == 0:
                print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
                fps = 30 / (time.time() - loopTime)
                loopTime = time.time()
                print("识别到的车牌数:\t", cnt_plate)
            # img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Inference
            # print('--> Running model')
            outputs = rknn1.inference(inputs=[img])
            # print('done')

            # post process
            input0_data = outputs[0]
            input1_data = outputs[1]
            input2_data = outputs[2]

            input0_data = input0_data.reshape([3, -1] + list(input0_data.shape[-2:]))
            input1_data = input1_data.reshape([3, -1] + list(input1_data.shape[-2:]))
            input2_data = input2_data.reshape([3, -1] + list(input2_data.shape[-2:]))

            input_data = list()
            input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
            input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
            input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

            boxes, classes, scores = yolov5_post_process(input_data)

            img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if boxes is not None:
                # draw(img_1, boxes, scores, classes)
                for box in boxes:
                    cnt_plate += 1
                    x1, y1, x2, y2 = box
                    x1 = int(x1)
                    x2 = int(x2)
                    y1 = int(y1)
                    y2 = int(y2)
                    cv2.rectangle(img_1, (x1, y1), (x2, y2), (0, 255, 0), 1, 1)
                    roi = img_1[y1:y2, x1:x2]
                    try:
                        roi = cv2.resize(roi, (168, 48))
                        output = rknn2.inference(inputs=[roi])
                        input_data = np.swapaxes(output[0], 1, 2)
                        index = np.argmax(input_data, axis=1)
                        plate_no = decodePlate(index)
                        img_1 = cv2ImgAddText(img_1, str(plate_no), x1, y1 - 30, (0, 255, 0), 30)
                        # print(plate_no)
                    except:
                        continue
            img = cv2.cvtColor(img_1, cv2.COLOR_RGB2BGR)
            cv2.putText(img_1, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        2)  # 在图像上显示帧率
            cv2.imshow("MIPI Camera", img_1)
        # 按下'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("总平均帧率\t", frames / (time.time() - initTime))
    """
    # show output
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rknn1.release()
    rknn2.release()
