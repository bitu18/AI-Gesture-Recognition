import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

import '@tensorflow/tfjs-converter';
import '@tensorflow/tfjs-backend-webgl';
// import '@tensorflow/tfjs-m';

import './App.css';
import { useEffect, useRef, useState } from 'react';


const TRAINING_TIMES = 50;
function App() {
  const [ready, setReady] = useState(false);
  const [showTitle, setShowTitle] = useState(true);
  const [showPercentTraining, setShowPercentTraining] = useState("");
  const [startPrediction, setStartPrediction] = useState(false);
  const [trainedLabels, setTrainedLabels] = useState(new Set());
  const [showIcon, setShowIcon] = useState(false);
  const [icon, setIcon] = useState("");

  const videoRef = useRef();
  const mobilenetModule = useRef();
  const classifier = useRef();

  const gestureIcons = {
    "none": "",
    "point_right": "👉",
    "point_left": "👈",
    "thumbs_up": "👍",
    "thumbs_down": "👎",
    "victory": "✌️",
    "ok": "👌",
    "heart": "❤️",
  };

  const init = async () => {
    await setupCamera();

    mobilenetModule.current = await mobilenet.load();
    classifier.current = knnClassifier.create();
    setReady(true);
  }

  const setupCamera = () => {
    return new Promise((resolve, reject) => {
      navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then((stream) => {
          videoRef.current.srcObject = stream;
          videoRef.current.addEventListener("loadeddata", resolve);
        })
        .catch(reject);
    });
  };

  const sleep = (ms = 0) => {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  const train = async (label) => {
    console.log(`Training for ${label}`);
    for (let i = 0; i < TRAINING_TIMES; i++) {
      // console.log(`Progress: ${parseInt((i + 1) * 100 / TRAINING_TIMES)}%`);
      setShowPercentTraining(`${parseInt((i + 1) * 100 / TRAINING_TIMES)}%`);
      await training(label);
    }

    // Mark this label as trained
    setTrainedLabels(prev => {
      const newSet = new Set(prev);
      newSet.add(label);

      if(newSet.size === Object.keys(gestureIcons).length) {
        setStartPrediction(true);
        setShowTitle(false);
        setReady(false);
      }

      return newSet;
    })
  }

  const training = (label) => {
    return new Promise(async (resolve) => {
      const embedding = mobilenetModule.current.infer(videoRef.current, true);
      classifier.current.addExample(embedding, label);
      await sleep(100);
      resolve();


    });
  }

  const handleStartPrediction = async() => {
     if (!classifier.current || !mobilenetModule.current) return;

    const embedding = mobilenetModule.current.infer(videoRef.current, true);

    const result = await classifier.current.predictClass(embedding);

    console.log('Label:', result.label, 'Confidence:', result.confidences[result.label]);

    setIcon(gestureIcons[result.label]);
    setShowIcon(true);

    await sleep(200);

    handleStartPrediction();
    setShowIcon(false);
  }

  useEffect(() => {
    init();

    // Cleanup function
    return () => { }

  }, []);

  return (
    <div className="main">
      <video ref={videoRef} className='video' autoPlay />

      <div className='control'>
        {ready ? ( <div className='btn--group'>
          <h2 className='title'>You can train the data</h2>
          <button className='btn' onClick={() => train("none")}>Training None</button>
          <button className='btn' onClick={() => train("point_right")}>Training 👉</button>
          <button className='btn' onClick={() => train("point_left")}>Training 👈</button>
          <button className='btn' onClick={() => train("thumbs_up")}>Training 👍</button>
          <button className='btn' onClick={() => train("thumbs_down")}>Training 👎</button>
          <button className='btn' onClick={() => train("victory")}>Training ✌️</button>
          <button className='btn' onClick={() => train("ok")}>Training 👌</button>
          <button className='btn' onClick={() => train("heart")}>Training ❤️</button>

          <h2 className='percentage'>{showPercentTraining}</h2>
        </div>) : 
         (showTitle && <h2 className='title'>We are setting up camera</h2>)
        }
       
        {startPrediction && <div className='btn--start'>
          <h2 className='title'>Start Prediction</h2>
          <button className='btn' onClick={handleStartPrediction}>Start</button>
        </div>}

         {showIcon && <div className='gesture'>{icon}</div>}
      </div>

      
    </div>
  );
}

export default App;
