import * as posenet from '@tensorflow-models/posenet';
import { drawSkeleton, drawKeypoints } from './demo_util';
import { saveAs } from 'file-saver';
import GIF from './gifjs/gif';

function loadImage(src) {
    return new Promise((resolve, reject) => {
        let img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
    });
}

function loadVideo(src) {
    return new Promise((resolve, reject) => {
        let vid = document.createElement("video");
        vid.autoplay = false;
        vid.onloadend = () => resolve(vid);
        vid.onerror = reject;
        vid.src = src;
    });
}

export async function start() {

    const framerate = 30;

    // Setup gif rendering
    var gif = new GIF({
        workers: 2,
        quality: 10
    });
    let gifRenderStartTime;
    gif.on('finished', function (blob) {
        const now = new Date();
        console.log(`Gif rendering took ${now - gifRenderStartTime}ms`);
        const dateFormatted = `${now.getUTCFullYear()}-${now.getUTCMonth()}-${now.getUTCDate()}_` +
            `${now.getUTCHours()}:${now.getUTCMinutes()}:${now.getUTCSeconds()}`;
        saveAs(blob, "pose_" + dateFormatted + ".gif");
    });

    const scaleFactor = 0.5;
    const flipHorizontal = false;
    const outputStride = 16;
    const videoElement = document.getElementById("video");


    // Draw image onto canvas
    const canvasElement = document.getElementById("canvas");
    canvasElement.width = videoElement.width;
    canvasElement.height = videoElement.height;
    const ctx = canvasElement.getContext("2d");

    // load the posenet model
    console.log("Loading network...");
    const timeBeforeLoad = new Date().getTime();
    const net = await posenet.load({
        architecture: 'ResNet50',
        outputStride: 32,
        inputResolution: 257,
        quantBytes: 2
    });
    console.log(`It took ${Math.round(new Date().getTime() - timeBeforeLoad)}ms to load`);
    videoElement.pause();
    const timeBeforePoseDetection = new Date().getTime();
    let timeAtFirstPoseDetection;
    let currentFrame = 0;

    const renderNextFrame = async () => {
        videoElement.pause();
        videoElement.currentTime += 1 / framerate;
        console.log("Estimating pose...");
        // ctx.drawImage(videoElement, 0, 0, videoElement.width, videoElement.height);
        const pose = await net.estimateSinglePose(videoElement, scaleFactor, flipHorizontal, outputStride);
        if (currentFrame === 0) {
            timeAtFirstPoseDetection = new Date().getTime();
            console.log(`First pose took ${timeAtFirstPoseDetection - timeBeforePoseDetection}ms`);
            console.log(pose);
        }

        const minPartConfidence = 0.1;

        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvasElement.width, canvasElement.height);

        drawKeypoints(pose.keypoints, minPartConfidence, ctx);
        drawSkeleton(pose.keypoints, minPartConfidence, ctx);

        gif.addFrame(canvasElement, { copy: true, delay: Math.round(1000 / framerate) });
        currentFrame++;

        if (videoElement.ended) {
            const now = new Date().getTime();
            console.log(`Post first pose detection took ${now - timeAtFirstPoseDetection}ms`);
            console.log(`Total pose detection took ${now - timeBeforePoseDetection}ms`);
            console.log("GIF Rendering started");
            gifRenderStartTime = new Date().getTime();
            gif.render();
        } else {
            renderNextFrame();
        }

    }

    renderNextFrame();

}

start();