<template>
  <div>
    <div class = "header">
      <img src="../assets/LB.png" alt="Logo" class="logo" />
      <h1>{{ msg }}</h1>
    </div>
    
    <div class="video-container">
      <div class="video">
        <video
          class="camera_video"
          ref="videoStream"
          width="400"
          height="300"
          autoplay
        ></video>
        <n-space>
          <n-button type="primary" @click="startCamera"> Camera-On </n-button>
        </n-space>
      </div>
      <div class="video">
        <img
          :src="('data:image/jpeg;base64,' + processedImage) || defaultImage"
          class="photo"
          alt=""
          width="400"
          height="300"
        />
        <n-space>
          <n-button type="error" @click="stopCamera"> Camera-Off </n-button>
        </n-space>
      </div>
    </div>

    <div class="footer">
      <n-button @click="sendMessage"> Check </n-button>
    </div>
  </div>
</template>

<script setup>
import io from "socket.io-client";
import { ref, onBeforeUnmount } from "vue";
import { useNotification, useMessage, NSpace, NButton } from "naive-ui";
// import axios from "axios";

const socket = io("http://localhost:5000", {
  extraHeaders: {
    "Access-Control-Allow-Origin": "http://localhost:8000",
  },
});

// 视频流
const videoStream = ref(null);
const processedImage = ref("");
const defaultImage = "../assets/AlonsoWindows.jpg";
// 计时器
let screenshotInterval;
// 标题
const msg = "RKNN";
// message
const message = ref("");

socket.on("connect", () => {
  console.log(socket.id);
  console.log("Connected to server");
});

socket.on("disconnect", () => {
  console.log(socket.id);
});

// 监听来自开发板的消息
socket.on("my_message", (data) => {
  console.log(data);
  if(data == "Server received your message")
    message.value = "Success";
  else
    message.value = "Fail";
});

// 监听来自开发板的截图
socket.on("processed", (img_back) => {
  console.log("Received processed image");

  if (typeof img_back === "object" && img_back instanceof ArrayBuffer) {
    // 如果是 ArrayBuffer，则将其转换为字符串
    const decoder = new TextDecoder("utf-8");
    img_back = decoder.decode(img_back);
  }

  processedImage.value = img_back;

  // 将 base64 编码的图像数据转换为 Blob 对象
  // const processedImageBlobValue = new Blob([base64ToArrayBuffer(img_back)], { type: "image/png" });
  // processedImage.value = URL.createObjectURL(processedImageBlobValue);
});

// 发送消息到服务器
const sendMessage = () => {
  socket.emit("my_message", "Hello, server!");
  // message.sendMessage(message.value);
};

// 截图并传输到服务器
const captureAndSend = () => {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  canvas.width = videoStream.value.videoWidth;
  canvas.height = videoStream.value.videoHeight;
  context.drawImage(videoStream.value, 0, 0, canvas.width, canvas.height);
  const imageData = canvas.toDataURL("image/jpeg").split(",")[1];

  /*
  // 发送 POST 请求 (async)
  try {
    const response = await axios.post('http://localhost:5000/process_image', { image: imageData });

    // 处理响应
    if (response.status === 200) {
      const processed_image_data = response.data.processed_image;
      if (processed_image_data) {
        console.log("Processed image received:", processed_image_data);
      } else {
        console.error("Error: No processed image data received");
      }
    } else {
      console.error("Error:", response.status);
    }
  } catch (error) {
    console.error("Error:", error.message);
  }
  */

  // 将截图发送到服务器
  console.log("Sending screenshot");
  socket.emit("screenshot", imageData);
};

// 调用摄像头
const startCamera = () => {
  if (navigator.mediaDevices) {
    navigator.mediaDevices
      .getUserMedia({ audio: false, video: true })
      .then((stream) => {
        // 将视频流传入video控件
        videoStream.value.srcObject = stream;
        // 播放
        videoStream.value.play();

        // 每隔一段时间执行截图和传输
        screenshotInterval = setInterval(() => {
          captureAndSend();
          console.log("capture");
        }, 80); // 每0.08秒截图一次
      })
      .catch((err) => {
        console.log(err);
      });
  } else {
    window.alert("该浏览器不支持开启摄像头，请更换最新版浏览器");
  }
};

const stopCamera = () => {
  let stream = videoStream.value.srcObject;
  if (!stream) return;
  let tracks = stream.getTracks();
  tracks.forEach((x) => {
    x.stop();
  });

  // 清除定时器
  clearInterval(screenshotInterval);
};

// 在组件销毁前停止摄像头
onBeforeUnmount(() => {
  stopCamera();
});

/* 
// 辅助函数，将 base64 编码的字符串转换为 ArrayBuffer
const base64ToArrayBuffer = (base64) => {
  const paddedBase64 = base64 + "=".repeat(4 - (base64.length % 4));
  const binaryString = window.atob(paddedBase64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; ++i) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
};
*/

</script>

<style scoped>
.logo {
  position: flex;
  top: 0;
  left: 0;
  max-width: 100px;
  margin: 10px;
}
.video-container {
  display: flex;
  justify-content: space-between;
  margin: 20px;
}
.video {
  text-align: center;
}
.camera_video {
  width: 800px;
  height: 600px;
  margin-bottom: 10px;
  padding: 10px;
}
.photo {
  width: 800px;
  height: 600px;
  margin-bottom: 0px;
  padding: 10px;
}
.footer {
  text-align: left;
  margin: 20px;
}
.header {
  display: flex;
  justify-content: space-between;
  margin: 20px;
}
</style>