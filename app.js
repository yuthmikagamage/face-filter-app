import * as THREE from "three";
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

const scene = new THREE.Scene();

const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 1000);

camera.position.z = 5;

const renderer = new THREE.WebGLRenderer({
  canvas: document.querySelector("#filter"),
  alpha: true,
  antialias: true,
});

renderer.setSize(streamCanvas.width, streamCanvas.height);
renderer.setClearColor(0x000000, 0);

const faceAnchor = new THREE.Object3D();

faceAnchor.rotation.order = "ZYX";
faceAnchor.visible = false;

scene.add(faceAnchor);

const geometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);

const material = new THREE.MeshNormalMaterial();

const cube = new THREE.Mesh(geometry, material);

faceAnchor.add(cube);

const getDistance = (a, b) => {
  return Math.hypot(a.x - b.x, a.y - b.y);
};

const getOrientation = (result, canvas) => {
  const data = result.keypoints;

  const pos_x = data[6].x - canvas.width / 2;
  const pos_y = (data[6].y - canvas.height / 2) * -1;

  const rot_x_a = getDistance(data[197], data[168]);
  const rot_x_b = getDistance(data[200], data[152]);

  const rot_x = Math.asin((0.5 - rot_x_b / (rot_x_a + rot_x_b)) * 2);

  const rot_y_a = getDistance(data[33], data[133]);
  const rot_y_b = getDistance(data[362], data[263]);

  const rot_y = Math.asin((0.5 - rot_y_b / (rot_y_a + rot_y_b)) * 2) * 2.5;

  const rot_z_y = data[33].y - data[263].y;
  const rot_z_d = getDistance(data[33], data[263]);

  const rot_z = Math.asin(rot_z_y / rot_z_d);

  const scale = getDistance(data[33], data[263]) * 0.007;

  return {
    position: [pos_x * 0.0029, pos_y * 0.0029, 0],
    rotation: [rot_x, rot_y, rot_z],
    scale: [scale, scale, scale],
  };
};

const update = async () => {
  streamContext.drawImage(video, 0, 0, streamCanvas.width, streamCanvas.height);

  const results = await detector.estimateFaces(streamCanvas, {
    flipHorizontal: false,
    predictIrises: false,
    maxFaces: 1,
  });

  if (results.length > 0) {
    const orientation = getOrientation(results[0], streamCanvas);

    faceAnchor.position.set(...orientation.position);

    faceAnchor.rotation.set(...orientation.rotation);

    faceAnchor.scale.set(...orientation.scale);

    faceAnchor.visible = true;
  } else {
    faceAnchor.visible = false;
  }

  renderer.render(scene, camera);

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
        aspectRatio: 1,
        width: { ideal: 720 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    video.srcObject = stream;

    await video.play();

    startBtn.style.display = "none";

    update();
  } catch (error) {
    console.error(error);
    alert("Unable to start app.");
  }
});
