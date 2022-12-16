let video;
let poseNet;
let pose;
let skeleton;

let brain;

let state = 'waiting';
let targetLabel;

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function keyPressed() {
	console.log(`key presssed`);
  if (key=='s') {
    brain.saveData();
		// brain.saveData("test");
  } else {
    targetLabel = key; //change here to other picture
    console.log(targetLabel);

		await delay(10000);
    console.log('collecting');
    state = 'collecting';

		await delay(10000);
    console.log('not collecting');
    state = 'waiting';
  }
}

function setup() {
	createCanvas(640, 480);
	video = createCapture(VIDEO);
	video.hide();
	poseNet = ml5.poseNet(video, modelLoaded);
	poseNet.on('pose', gotPoses);

	let options = {
		inputs: 34,
		outputs: 4,
		task: 'classification',
		debug: true
	}
	brain = ml5.neuralNetwork(options);
	const modelInfo = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin',
	}
	brain.load(modelInfo, brainLoaded)
}

function brainLoaded() {
	console.log(`classification model ready!`);
	// brain.classify(inputs, gotResults);
	classifyPose()
}

function dataReady() {
	brain.normalizeData()
	brain.train({epochs: 40}, finished);
}

function finished() {
	console.log(`model trained`);
	brain.save();
}

function gotPoses(poses) {
	// console.log(poses);
	if (poses.length > 0) {
		pose = poses[0].pose;
		skeleton = poses[0].skeleton;

		if (state == 'collecting') {

			let inputs = [];
			for (let i = 0; i < pose.keypoints.length; i++) {
				let x = pose.keypoints[i].position.x;
				let y = pose.keypoints[i].position.y;
				// should inputs be a flattened array or 2d array?
				inputs.push(x);
				inputs.push(y);
			}
			let target = [targetLabel];
			brain.addData(inputs, target);
		}
	}
}

function modelLoaded() {
	console.log('poseNet ready');
	// classifyPose()
}

function classifyPose() {
	if (pose) {
		let inputs = [];
		for (let i = 0; i < pose.keypoints.length; i++) {
			let x = pose.keypoints[i].position.x;
			let y = pose.keypoints[i].position.y;
			// should inputs be a flattened array or 2d array?
			inputs.push(x);
			inputs.push(y);
		}
		brain.classify(inputs, gotResult)
	} else {
		setTimeout(classifyPose, 100);
	}
}

function gotResult(err, results) {
	console.log(results);
	console.log(results[0].label);
}

function draw() {
	translate(video.width, 0);
	scale(-1, 1);
	image(video, 0, 0, video.width, video.height);


	if (pose) {
		fill(255,0,0);
		ellipse(pose.nose.x, pose.nose.y, 50);
		fill(0,0,225);
		ellipse(pose.leftWrist.x, pose.leftWrist.y, 30);
		ellipse(pose.rightWrist.x, pose.rightWrist.y, 30);

		for (let i = 0; i < pose.keypoints.length; i++) {
			let x = pose.keypoints[i].position.x;
			let y = pose.keypoints[i].position.y;
			fill(0,255,0);
			ellipse(x,y, 16, 16);
		}

		for (let i = 0; i < skeleton.length; i++) {
      let a = skeleton[i][0];
      let b = skeleton[i][1];
      strokeWeight(3);
      stroke(255);
      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }
	}
}
