//network specific
let avgcost = 0;
let gen = 1;
let avgperc = 1;
let MNIST = [];
let notMNIST;
let humanarr = [];
let writemode = false;
let scalar = 15;
let grams = 2;

//adjustables
//gpt?
let learningset = 1;//context length
let heads = 3;
let querykeydim = 5;//size of smaller dimensional query-key-valuedown space
let encodesize = 10;//value up, encoder vector size full representation
let sampleset = 200;//sample data
let smtemp = 1;//temperature of softmax algorithm

//model parameters
let layers = 3;//hidden layers
let hiddensize = 40;
let inputsize = 784;
let outputsize = 2;

//training
let wi = 1;//weight initialization values
let trialspersesh = 40;
let scale = 2;//activation scale
let learningrate = 0.2;//velocity of adjust

//other
let outputarr = [0,1];
let addbias = true;
let type = "sigmoid";

function setup() {

	createCanvas(windowWidth,windowHeight);
	background(0);
	loadshitlinear();
	
	humanarr = maketensor(1,[inputsize],0);
	for (a = 0; a < notMNIST.rows.length; a++) {
		let q = notMNIST.rows[a].arr;
		MNIST[a] = [];
		for (b = 0; b < q.length; b++) {
			MNIST[a][b] = int(q[b]); 
		}
	}
	MNIST = Bsort(MNIST,0,false,false,true,0);
	
}

function preload() {
	notMNIST = loadTable('MNIST.csv', 'csv', 'header');
}

function draw() {
	
	if (keyCode == SHIFT) {
		keyCode = SHIFT;
		if (writemode == true) {
			writemode = false;
			humanarr = maketensor(1,[inputsize],0);
		}
		for (cc = 0; cc < trialspersesh; cc++) {
			trainlinear(cc);
		}
	}//train
	else if (keyCode == CONTROL) {
		writemode = true;
		if (mouseX < 420 && mouseY < 420 && mouseIsPressed) {
			humanarr[(mouseX/scalar-0.5)+sqrt(inputsize)*(mouseY/scalar-0.5)] = 
				255-humanarr[humanarr[(mouseX/scalar-0.5)+sqrt(inputsize)*(mouseY/scalar-0.5)]];
		}
		getnetguess(humanarr);
	}//writemode

}

function getnetguess(input) {

	//load data
	let trainingIs = div2d([CA(input)],maketensor(2,[1,inputsize],255))[0];
	
	//gets network's best guess
	let holdmyarr = runlinear(trainingIs,layers+1,weights,biases);//assigns networks best guess
	let networkarr = Bsort(CA(holdmyarr[0]),outputarr,false,true);
	
	//various ui components and cost stuff
	stroke(0)
	fill(0,0,0);	
	rect(0,0,windowWidth,windowHeight)
	
	//draw box
	for (a = 0; a < sqrt(inputsize); a++) {
		for (b = 0; b < sqrt(inputsize); b++) {
			fill(trainingIs[a*sqrt(inputsize)+b]*255);
			stroke(255);
			strokeWeight(3);
			rect(scalar*b,scalar*a,scalar,scalar)
		}
	}
	
	//draw guess
	textSize(30)
	fill(255)
	stroke(0)
	text("guess: " + networkarr[0][0] + "\n" + 
			 "confidence: " + networkarr[0][1],7,430);
	
}

function screeninfo(networkarr,correctcheck,totalcost,trainingIs) {
	
	//various ui components and cost stuff
	stroke(0)
	fill(0,0,0);	
	rect(0,0,windowWidth,windowHeight)
	
	//draw box
	for (a = 0; a < sqrt(inputsize); a++) {
		for (b = 0; b < sqrt(inputsize); b++) {
			fill(trainingIs[a*sqrt(inputsize)+b]);
			stroke(255);
			strokeWeight(1);
			rect(scalar*b,scalar*a,scalar,scalar)
		}
	}
	
	//update info
	gen++;
	avgcost = (avgcost*(gen-1)+totalcost)/gen;
	avgperc = (avgperc*(gen-1)+int(networkarr[0][0]==correctcheck))/gen;
	
	//draw info
	textSize(28)
	fill(255*(1-int(networkarr[0][0]==correctcheck)),255*int(networkarr[0][0]==correctcheck),0)
	stroke(0)
	text("correct: " + correctcheck + "\n" +
			 "guess: " + networkarr[0][0] + "\n" + 
			 "confidence: " + round(networkarr[0][1]*100)/100 + "\n" + 
			 "avg cost: " + round(avgcost*100)/100 + "\n" + 
			 "avg correct%: " + round(avgperc*10000)/100 + "%\n" + 
			 "generation: " + gen,7,450);
	
}

function runexample(input,label,cc) {
	
	//load data
	let trainingIs = div2d([CA(input)],maketensor(2,[1,inputsize],255))[0];
	
	//gets network's best guess
	let holdmyarr = runlinear(trainingIs,layers+1,weights,biases);//assigns networks best guess
	let networkarr = Bsort(CA(holdmyarr[0]),outputarr,false,true);
	let networkguess = networkarr[0][0];
	neuronstore = holdmyarr[1];
	
	//cost calc
	let totalcost = 0;
	let costarr = [];
	for (a = 0; a < networkarr.length; a++) {
		costarr[networkarr[a][0]] = int(networkarr[a][0]==label)-networkarr[a][1];
		totalcost += abs(costarr[networkarr[a][0]]);
	}
	
	if (cc == trialspersesh-1) {
		screeninfo(networkarr,label,totalcost,CA(input));
	}
	return costarr;
	
}

function keyReleased() {
	
  if (keyCode == SHIFT) {
    keyCode = "";
	}
	return false;
	
}
