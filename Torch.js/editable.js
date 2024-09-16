//network specific

//  /X\ indicates it changes function name- need docs

//ADJUSTABLES
//linear
let wi = 1;//weight initialization values
let layers = 0;//hidden layers
let hiddensize = 0;//maximum hidden layer size
let inputsize = 0;//max input size
let outputsize = 0;//max output size
let addbias = false;//add bias values to network

//gpt
let traindat = 0.9;//percentage of data used for gpt training
let learningset = 0;//context length
let heads = 0;//heads of gpt network
let querykeydim = 0;//size of smaller dimensional query-key-valuedown space (less then encodesize)
let encodesize = 0;//value up, encoder vector size full representation
let sampleset = 0;//sample data
let smtemperature = 1;//temperature of softmax algorithm
let sureness = 0;//gpt training value
let ffnlayers = 0;//gpt second network layers
let convthresh = 0;//convergence threshold (0-2)

//adamW
let alpha = 0.001;
let b1 = 0.9;
let b2 = 0.999;
let epsilon = 0.000001;

//generative
let newlayer = 0.2;//chance of making new layer in gen
let weightmult = 2;//weights:neurons+biases ratio
let iterations = 0;//models on screen
let timealive = 0;//how long each training epoch lasts

//general
let trialspersesh = 0;//various uses, used for averaging results
let learningrate = 0.5;//velocity of training adjustments
let type = "sigmoid";//type of activation function (sigmoid,RELU,GELU)
let scale = 1;//normal dist of activation functions

//misc
let textbox;
let lines;
let e = 2.718281828459045;
let pi = 3.14159265358979;

function screeninfo(totalcost,label,data,netarr) {
	
	//write some stuff for screen
	
}

function preload() {
	//get some file info
}

function setup() {
	
	createCanvas(windowWidth, windowHeight);
	background(0);
	textAlign(CENTER);
	text(windowWidth/2,windowHeight/2,
	     "Welcome to Torch.js."+
		 "\nThis is currently being loaded from the default setup() function."+
		 "\nPlease go to editable.js to implement code");
	
	//load some data

	//loadshit /something\ ();
	
}

function draw() {
	
	keyPressed();

}

function keyReleased() {
  if (keyCode == SHIFT) {
    keyCode = "";
		return false;
  }
}//press and hold logic

function keyPressed() {
	
	if (keyCode == SHIFT) {
		//call training
	}
	
  return false;

}//train with holding shift

function runexample() {
//load some data
		
	let netarr = /*run/model\(data)*/;
	let totalcost = 0;
	for (a = 0; a < netarr.length; a++) {
		if (netarr[a][0] != label) {
			totalcost += netarr[a][1];
			costpertoken[layers][netarr[a][0]] += -1*sureness*netarr[a][1];
		}
		else {
			totalcost += 1-netarr[a][1];
			costpertoken[layers][netarr[a][0]] += correctness*(1-netarr[a][1]);
		}
	}//costcalc
	screeninfo(/*totalcost,label,data,netarr*/)//draw stuff

	return /*variable*/

}
function getNetGuess() {
	
	//load some data for the network from user
	
	//draw stuff on screen
	
}
