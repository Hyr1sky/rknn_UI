import eventlet
import socketio
import base64
import cv2
import os
import time
import numpy as np
from flask import Flask, request, jsonify

sio = socketio.Server(cors_allowed_origins="http://localhost:8000")
app = socketio.WSGIApp(sio)
script_dir = os.path.dirname(os.path.realpath(__file__))
# flask_app = Flask(__name__)

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
    # data 是 Base64
    img_decoded = base64.b64decode(data)
    img_array = np.frombuffer(img_decoded, np.uint8)
    print("Received image data length:", len(img_array))
    # 解码图像
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # 在这里处理图片
    if img is None:
        print("Error: Decoded image is None")
        img_back = base64.b16encode(img_decoded).decode('utf-8')
        sio.emit('processed', img_back, sid)
    else:
        print("Image decoded successfully")
        # 例如，翻转图像
        img_flipped = cv2.flip(img, 1)  # 1表示水平翻转，0表示垂直翻转，-1表示同时水平和垂直翻转

        # 编码处理后的图像
        img_back = base64.b64encode(cv2.imencode('.png', img_flipped)[1])
        print("Image processed and encoded successfully")
        print("Sending processed image data length:", len(img_back))
        # 发送处理后的图像数据回客户端
        sio.emit('processed', img_back, sid)

"""
@app.route('/process_image', methods=['POST'])
def process_image():
    # 获取 POST 请求中的 JSON 数据
    data = request.get_json()

    # 从数据中提取图像，这取决于你的数据结构
    image_data = data.get('image', None)

    if image_data:
        # 解码图像
        img_decoded = base64.b64decode(image_data)
        img_array = np.frombuffer(img_decoded, np.uint8)

        # 在这里处理图像
        if img_array is not None:
            # 例如，翻转图像
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img_flipped = cv2.flip(img, 1)

            # 编码处理后的图像
            _, processed_encoded = cv2.imencode('.png', img_flipped)
            processed_image_data = base64.b64encode(processed_encoded).decode('utf-8')

            # 返回处理后的结果
            result = {'processed_image': processed_image_data}
            return jsonify(result)
        else:
            return jsonify({'error': 'Error decoding image'}), 500
    else:
        return jsonify({'error': 'No image data received'}), 400
"""

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)