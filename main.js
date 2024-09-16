//network specific
let avgcost = 1;
let gen = 1;
let avgperc = 0;
let MNIST = [];
let notMNIST;
let humanarr = [];
let writemode = false;
let scalar = 15;
let grams = 2;
let chartarrs = [];

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
let hiddensize = 80;
let inputsize = 784;
let outputsize = 2;

//training
let wi = 1;//weight initialization values
let trialspersesh = 1;//runs before data
let scale = 3;//activation scale
let learningrate = 0.2;//velocity of adjust
let correctness = 0.2;//strength of cost

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
	chartarrs = [[avgcost,avgperc]];
	
}

function preload() {
	notMNIST = loadTable('MNISTreal.csv', 'csv', 'header');
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
		if (mouseX < scalar*sqrt(inputsize) && mouseY < scalar*sqrt(inputsize) && mouseIsPressed == true) {
			let loc = round(mouseX/scalar-0.5)+sqrt(inputsize)*round(mouseY/scalar-0.5);
			humanarr[loc] = 
				255-humanarr[humanarr[loc]];
		}
		//runexample(humanarr,"writing",trialspersesh-1);
		let rline = MNIST[rr(0,MNIST.length/10*grams)];
		runexample(rline.slice(1,inputsize+1),rline[0],0);//output
	}//writemode

}

function screeninfo(networkarr,correctcheck,totalcost,trainingIs,cc) {
		
	//update info
	if (correctcheck !== "writing") {
		gen++;
		avgcost = (avgcost*(gen-1)+totalcost)/gen;
		avgperc = (avgperc*(gen-1)+int(networkarr[0][0]==correctcheck))/gen;
		chartarrs[gen-1] = [avgcost,avgperc];
	}
	if (cc != trialspersesh-1) {
		return;
	}
		
	stroke(0)
	fill(0,0,0);	
	rect(0,0,windowWidth,windowHeight)
	
	//draw charts
	let gaps = (windowWidth-460)/gen;
	fill(255);
	stroke(255);
	for (a = 0; a < gen-2; a++) {
		line(a*gaps+440,200-chartarrs[a][0]/outputsize*200,(a+1)*gaps+440,200-chartarrs[a+1][0]/outputsize*200)
	}
	for (a = 0; a < gen-2; a++) {
		line(a*gaps+440,200-chartarrs[a][1]/outputsize*200+300,(a+1)*gaps+440,200-chartarrs[a+1][1]/outputsize*200+300)
	}
	
	//draw box
	for (a = 0; a < sqrt(inputsize); a++) {
		for (b = 0; b < sqrt(inputsize); b++) {
			fill(trainingIs[a*sqrt(inputsize)+b]);
			stroke(255);
			strokeWeight(1);
			rect(scalar*b,scalar*a,scalar,scalar)
		}
	}
	
	//draw info
	textSize(24)
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
		let costval = int(networkarr[a][0]==label)-networkarr[a][1];
		costarr[networkarr[a][0]] = correctness*costval;
		totalcost += abs(costval);
	}
	
	screeninfo(networkarr,label,totalcost,CA(input),cc);
	return costarr;
	
}

function keyReleased() {
	
  if (keyCode == SHIFT) {
    keyCode = "";
	}
	return false;
	
}
