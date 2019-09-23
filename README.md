# PoseNet to GIF

Uses:
- TensorFlow PoseNet (https://github.com/tensorflow/tfjs-models/tree/master/posenet)
- gif.js (https://github.com/jnordberg/gif.js)

To take a video and generate a gif with the skeleton and keypoints of any body poses detected in the video.

## How to install and run
To install run
```
yarn 
```

Before running copy a video called test2.mp4 into a videos/ folder in the repo

In order to run
```
yarn start
```

Then go to http://localhost:1234 where your video will be shown along side the skeleton rendering

In the Javascript console you can the time taken and progress of the pose detection