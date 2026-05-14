import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";

import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";

await tf.setBackend("webgl");

const video = document.querySelector("#video");
const startBtn = document.querySelector("#startBtn");

video.muted = true;
video.autoplay = true;
video.playsInline = true;

const streamCanvas = document.querySelector("#stream");

const streamContext = streamCanvas.getContext("2d", {
  willReadFrequently: true,
});

streamContext.translate(streamCanvas.width, 0);
streamContext.scale(-1, 1);

let detector;

const update = async () => {
  streamContext.drawImage(video, 0, 0);

  const results = await detector.estimateFaces(streamCanvas, {
    flipHorizontal: false,
    predictIrises: false,
    maxFaces: 1,
  });

  if (results.length > 0) {
    console.log("Face detected:", results[0]);
  }

  requestAnimationFrame(update);
};

startBtn.addEventListener("click", async () => {
  try {
    detector = await faceLandmarksDetection.createDetector(
      faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
      {
        runtime: "tfjs",
      },
    );

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: 1080,
        height: 1080,
      },
      audio: false,
    });

    video.srcObject = stream;

    startBtn.style.display = "none";

    update();
  } catch (error) {
    console.error(error);
    alert("Unable to start app.");
  }
});
