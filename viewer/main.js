import * as THREE from 'three';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';

// Create a websocket connection to the server
const socket = new WebSocket('ws://localhost:8765');

const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
const controls = new PointerLockControls( camera, document.body );

// Read the base64 encoded image data from the server
// and set it to the background of the webpage
socket.addEventListener('message', (event) => {
  const data = JSON.parse(event.data);
  const image = new Image();
  image.src = 'data:image/jpeg;base64,' + data.image;
  image.onload = () => {
    document.body.style.backgroundImage = `url(${image.src})`;
  };
})

socket.addEventListener('open', () => {
  // Screen aspect ratio
  const aspect = window.innerWidth / window.innerHeight;

  // When the socket is open, send information about the camera
  // to the server (the FOV, aspect ratio, near and far clipping planes)
  const message = {
    type: 'cameraInfo',
    position: camera.position.toArray(),
    quat: camera.quaternion.toArray(),
    fovX: camera.fov,
    fovY: camera.fov,
    near: camera.near,
    far: camera.far,
    aspectRatio: aspect,
  };
  socket.send(JSON.stringify(message));
});

function lockControls() {
  controls.lock();
}
function unlockControls() {
  controls.unlock();
}

function sendCameraTransform() {
  // Round the position and rotation to 2 decimal places (toFixed is wrong)
  let position = camera.position.toArray().map((x) => Number(x.toFixed(4)));
  let quaternion = camera.quaternion.toArray().map((x) => Number(x.toFixed(4)));

  // Correct the position using the axis flipped quaternion (not the rotation)
  quaternion = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI).multiply(new THREE.Quaternion(...quaternion)).toArray();
  position = new THREE.Vector3(...position).applyQuaternion(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI)).toArray();

  // Screen aspect ratio
  const aspect = window.innerWidth / window.innerHeight;

  const message = {
    type: 'renderRequest',
    position: position,
    quat: quaternion,
    aspectRatio: aspect,
  };
  console.log(message);
  socket.send(JSON.stringify(message));
}

// When "w" or up arrow is pressed, move the camera forward
// When "s" or down arrow is pressed, move the camera backward.
document.addEventListener('keydown', (event) => {
  if (event.key === 'w' || event.key === 'ArrowUp') {
    // Move in the look direction of the camera
    const direction = new THREE.Vector3();
    camera.getWorldDirection(direction);
    direction.multiplyScalar(0.1);
    camera.position.add(direction);
    sendCameraTransform();
  }
  if (event.key === 's' || event.key === 'ArrowDown') {
    // Move in the opposite direction of the look direction of the camera
    const direction = new THREE.Vector3();
    camera.getWorldDirection(direction);
    direction.multiplyScalar(-0.1);
    camera.position.add(direction);
    sendCameraTransform();
  }
  if (event.key === 'a' || event.key === 'ArrowLeft') {
    // Move in the left direction of the camera
    const direction = new THREE.Vector3();
    camera.getWorldDirection(direction);
    direction.cross(new THREE.Vector3(0, 1, 0));
    direction.multiplyScalar(-0.1);
    camera.position.add(direction);
    sendCameraTransform();
  }
  if (event.key === 'd' || event.key === 'ArrowRight') {
    // Move in the right direction of the camera
    const direction = new THREE.Vector3();
    camera.getWorldDirection(direction);
    direction.cross(new THREE.Vector3(0, 1, 0));
    direction.multiplyScalar(0.1);
    camera.position.add(direction);
    sendCameraTransform();
  }
  if (event.key === 'q') {
    // Move down
    camera.position.y -= 0.1;
    sendCameraTransform();
  }
  if (event.key === 'e') {
    // Move up
    camera.position.y += 0.1;
    sendCameraTransform();
  }
})

// Listen to pointer lock control "change" event
// to detect when the user has locked or unlocked the controls
controls.addEventListener('change', () => {
  sendCameraTransform();
})



// Default listeners
document.addEventListener('mousedown', lockControls);
document.addEventListener('mouseup', unlockControls);
