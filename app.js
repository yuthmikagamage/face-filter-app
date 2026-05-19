import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
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
const camera = new THREE.OrthographicCamera();
const renderer = new THREE.WebGLRenderer({
  canvas: document.querySelector("#filter"),
  alpha: true,
  antialias: true,
});
renderer.setSize(streamCanvas.width, streamCanvas.height);
renderer.setClearColor(0x000000, 0);
camera.position.set(0, 0, 45);
scene.add(camera);

const faceAnchor = new THREE.Object3D();
faceAnchor.rotation.order = "ZYX";
faceAnchor.visible = false;
scene.add(faceAnchor);

const loader = new GLTFLoader();
loader.load(
  "assets/models/filters/model_1.glb",
  (gltf) => {
    faceAnchor.add(gltf.scene);
  },
  undefined,
  (err) => console.error(err),
);

const getDistance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

const getOrientation = (result, canvas) => {
  const data = result.keypoints;
  const pos_x = data[6].x - canvas.width / 2;
  const pos_y = (data[6].y - canvas.height / 2) * -1;
  const pos_z = 0;
  const rot_x_a = getDistance(data[197], data[168]);
  const rot_x_b = getDistance(data[200], data[152]);
  const rot_x = Math.asin((0.5 - rot_x_b / (rot_x_a + rot_x_b)) * 2);
  const rot_y_a = getDistance(data[33], data[133]);
  const rot_y_b = getDistance(data[362], data[263]);
  const rot_y = Math.asin((0.5 - rot_y_b / (rot_y_a + rot_y_b)) * 2) * 2.5;
  const rot_z_y = data[33].y - data[263].y;
  const rot_z_d = getDistance(data[33], data[263]);
  const rot_z =
    data[33].x < data[263].x
      ? Math.asin(rot_z_y / rot_z_d)
      : 1 - Math.asin(rot_z_y / rot_z_d) + Math.PI * 0.68;
  const scale = getDistance(data[33], data[263]) * 0.005;
  if (rot_y > 0.7 || rot_y < -0.7) {
    return null;
  }
  return {
    position: [pos_x * 0.0029, pos_y * 0.0029, pos_z],
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
    if (orientation) {
      faceAnchor.position.set(...orientation.position);
      faceAnchor.rotation.set(...orientation.rotation);
      faceAnchor.scale.set(...orientation.scale);
      faceAnchor.visible = true;
    } else {
      faceAnchor.visible = false;
    }
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
      { runtime: "tfjs" },
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
