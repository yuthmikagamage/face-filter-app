import "@tensorflow/tfjs-backend-webgl";

const video = document.querySelector("#video");

video.muted = true;
video.autoplay = true;
video.playsInline = true;

const stream = await navigator.mediaDevices.getUserMedia({
  video: {
    facingMode: "user",
    width: 1080,
    height: 1080,
  },
  audio: false,
});

video.srcObject = stream;

const streamCanvas = document.querySelector("#stream");

const streamContext = streamCanvas.getContext("2d");

streamContext.translate(streamCanvas.width, 0);
streamContext.scale(-1, 1);

const update = () => {
  streamContext.drawImage(video, 0, 0);

  requestAnimationFrame(update);
};

update();
