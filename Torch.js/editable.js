//network specific
let avgcost = 1;
let gen = 1;
let avgperc = 0;
let MNISTarr = [];
let notMNIST;
let humanarr = [];
let writemode = false;
let scalar = 15;
let grams = 10;
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
let outputsize = 10;

//training
let wi = 1;//weight initialization values
let scale = 3;//activation scale
let learningrate = 0.2;//velocity of adjust
let correctness = 0.2;//strength of cost

//other
let outputarr = [0,1,2,3,4,5,6,7,8,9];
let addbias = true;
let type = "sigmoid";

function setup() {

	createCanvas(windowWidth,windowHeight);
	background(0);
	loadshitlinear();
	
	humanarr = maketensor(1,[inputsize],0);
 	for (a = 0; a < notMNIST.rows.length; a++) {
		let q = notMNIST.rows[a].arr;
		MNISTarr[a] = [];
		for (b = 0; b < q.length; b++) {
			MNISTarr[a][b] = int(q[b]); 
		}
	}
	MNISTarr = Bsort(MNISTarr,0,false,false,true,0);
	chartarrs = [[avgcost,avgperc]];
	
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
		
		let rline = MNISTarr[rr(0,MNISTarr.length/10*grams)];
		let ink = rline.slice(1,inputsize+1);
		let labelk = rline[0];
		trainlinear(ink,labelk);
	}//train
	else if (keyCode == CONTROL) {
		writemode = true;
		if (mouseX < scalar*sqrt(inputsize) && mouseY < scalar*sqrt(inputsize) && mouseIsPressed == true) {
			let loc = round(mouseX/scalar-0.5)+sqrt(inputsize)*round(mouseY/scalar-0.5);
			humanarr[loc] = 
				255-humanarr[humanarr[loc]];
		}
		runexample(humanarr,"writing");
	}//writemode

}

function screeninfo(networkarr,correctcheck,totalcost,trainingIs) {
		
	//update info
	if (correctcheck !== "writing") {
		gen++;
		avgcost = (avgcost*(gen-1)+totalcost)/gen;
		avgperc = (avgperc*(gen-1)+int(networkarr[0][0]==correctcheck))/gen;
		chartarrs[gen-1] = [avgcost,avgperc];
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

function runexample(input,label) {
	
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
	
	screeninfo(networkarr,label,totalcost,CA(input));
	return costarr;
	
}

function keyReleased() {
	
  if (keyCode == SHIFT) {
    keyCode = "";
	}
	return false;
	
}
