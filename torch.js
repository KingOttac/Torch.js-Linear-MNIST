//dec

//arrays
let trainingIs = [];//used for storing different context sets
let costpertoken = [];//assigned by neuron number, represents derivative of cost function
let convertedlines = [];//giant array for dataset
let encoders = [];//vectors to encode tokens
let decoders = [];//vectors to decode tokens
let key = [];
let query = [];
let value = [];
let returns = [];
let biases = [];
let tokens = [];
let costarr = [];
let weights = [];
let neuronstore = [];

//general
let e = 2.718281828459045;
let pi = 3.14159265358979;
let inf = 2E+300;

function loadshitlinear() {

	weights = shapenet([inputsize,hiddensize,outputsize],false,2,layers,0,true,-wi,wi);
	biases = shapenet([hiddensize,hiddensize,outputsize],false,1,layers,0,true,-wi,wi);

}

function runlinear(input,qlayers,allweights,allbiases) {
	
	function linear(ARR,weightsarr,biasesarr) {

		let returnarr = matrixmult([ARR],weightsarr)[0];
		if (addbias == true) {
			returnarr = add2d([returnarr],[biasesarr])[0];
		}

		return returnarr;

	}//takes in 1d array and returns one transform with weights from layer + bias

	let nsra = [];
	nsra[0] = CA(input);
	let ra = activate(CA(input));
	for (a = 0; a < qlayers; a++) {
		nsra[a+1] = linear(CA(ra),allweights[a],allbiases[a]);
		ra = activate(CA(nsra[a+1]))
	}
	return [ra,nsra];

}//takes in parameters for layers (corr to weights), if sorted, input 
//returns an array of [output , each layer unactivated arr]

function trainlinear(cc) {
	
	let rline = MNIST[rr(0,MNIST.length/10*grams)];
	costpertoken = maketensor(2,[layers+1,hiddensize],0,false);//hidden
	costpertoken[layers] = runexample(rline.slice(1,inputsize+1),rline[0],cc);//output
	for (bb = layers; bb >= 0; bb--) {//layer
		for (aa = 0; aa < weights[bb].length; aa++) {//first neuron
			for (aa1 = 0; aa1 < weights[bb][aa].length; aa1++) {//second neuron
				let gfd = getfuncderiv(neuronstore[bb+1][aa1]);
				weights[bb][aa][aa1] += //sum of
					activate([neuronstore[bb][aa]])[0] * //in terms of zl- prev neuron is what influences zl
					gfd * //in terms of al- derivative of relu w/ respect to zl
					costpertoken[bb][aa1] *  //in terms of cost- desired change to cost
					learningrate;
				if (bb != 0) {
					biases[bb-1][aa] += 
						1 * //in terms of zl- bias does not influence zl
						gfd * //in terms of al- derivative of prev w/ respect to zl
						costpertoken[bb][aa1] *  //in terms of cost- desired change to cost down the line
						learningrate;
					costpertoken[bb-1][aa] += 
						weights[bb][aa][aa1] * //in terms of zl- weight is what influences zl
						gfd * //in terms of al- derivative of relu w/ respect to zl
						costpertoken[bb][aa1];  //in terms of cost- desired change to cost down the line
				}//next costs if not final layer
			}
		}
	}

}//train the model

function activate(ARR) {
	
	let ra = [];
	for (g = 0; g < ARR.length; g++) {
		if (type == "sigmoid") {
			ra[g] = sigmoid(ARR[g]);
		}
		else if (type == "RELU") {
			ra[g] = RELU(ARR[g]);
		}
		else if (type == "GELU") {
			ra[g] = GELU(ARR[g]);
		}
	}
	return ra;
	
}

function getfuncderiv(input) {
		
	if (type == "sigmoid") {
		return (pow(e,-1*input))/pow(1+pow(e,-1*input),2);
	}
	else if (type == "RELU") {
		return 1;
	}
	else if (type == "GELU") {
		let ndx = pow(e,1.702*input);
		return (ndx*(1+ndx+1.702*input))/pow(1+ndx,2)
	}

}

function untoken(Q) {
	
	for (g = 0; g < tokens.length; g++) {
		if (tokens[g] == Q) {
			return g;
		}
	}
	if (Q == "") {
		return 8;
	}//newline catch
	return -1;
	
}

function positioners(x) {
	
	let sx = learningset/1.57079633;
	return sin(x/sx);
	
}

function softmax(ARR) {
	
	let exsum = 0;
	let arrtoreturn = [];
	for (g = 0; g < ARR.length; g++) {
		exsum += pow(e,ARR[g]/smtemperature);
	}
	for (g = 0; g < ARR.length; g++) {
		arrtoreturn[g] = pow(e,ARR[g]/smtemperature)/exsum;
	}
	return arrtoreturn;
	
}//one dimentional note

function tril(ARR,v) {
	
	for (g = 0; g < ARR.length; g++) {
		for (g1 = 0; g1 < ARR[0].length; g1++) {
			if (g1 > g) {
				ARR[g][g1] = v;
			}
		}
	}
	
	return ARR;
	
}//create triangle of (v) values in 2d array

function transpose(ARR) {
	
	return ARR[0].map((_, colIndex) => ARR.map(row => row[colIndex]));
	
}//switch columns with rows, preserve values

function matrixmult(ARR1,ARR2) {
	
	if (ARR1[0].length != ARR2.length) {
		print("ERROR:\ninfo:\nARR1:",ARR1,"ARR2:",ARR2)
		exit()
	}
	let returnarr = [];
	for (g = 0; g < ARR1.length; g++) {
		returnarr[g] = [];
		for (g1 = 0; g1 < ARR2[0].length; g1++) {
			returnarr[g][g1] = 0;
		}
	}//initialize returnarrs
	
	for (g = 0; g < returnarr.length; g++) {
		for (g1 = 0; g1 < returnarr[g].length; g1++) {
			for (g2 = 0; g2 < ARR1[0].length; g2++) {
				returnarr[g][g1] += ARR1[g][g2]*ARR2[g2][g1];
			}
		}
	}
	
	return returnarr;
	
}//can also be used with ([vec1],transpose([vec2])) for dot product

function Bsort(ARR,sourceARR,softmaxb,highlow,byprop,prop) {
	
	if (softmaxb == true) {
		ARR = softmax(ARR);
	}
	
	let sorted = [];
	if (byprop != true) {
		sorted = transpose([sourceARR,ARR]);
	}
	else {
		sorted = ARR;
	}
	
	let arrp = 1;
	if (byprop == true) {
		arrp = prop;
	}
	if (highlow == true) {
		sorted.sort(function(p, q) {
			return q[arrp] - p[arrp];
		});
	}
	else {
		sorted.sort(function(p, q) {
			return p[arrp] - q[arrp];
		});
	}

	return sorted;
	
}//[0] is decoded value, [1] is strength of that value

function maketensor(dim,shapeARR,fill,ifrand,randl,randh,ifroundrand,ascending) {
	
	let ra = []
	for (g = 0; g < shapeARR[0] && dim > 0; g++) {
		ra[g] = [];
		for (g1 = 0; g1 < shapeARR[1] && dim > 1; g1++) {
			ra[g][g1] = [];
			for (g2 = 0; g2 < shapeARR[2] && dim > 2; g2++) {
				ra[g][g1][g2] = [];
				for (g3 = 0; g3 < shapeARR[3] && dim > 3; g3++) {
					ra[g][g1][g2][g3] = [];
					for (g4 = 0; g4 < shapeARR[4] && dim > 4; g4++) {
						ra[g][g1][g2][g3][g4] = [];
						for (g5 = 0; g5 < shapeARR[5] && dim > 5; g5++) {
							ra[g][g1][g2][g3][g4][g5] = getfill(g5);
						}
						if (dim == 5) {
							ra[g][g1][g2][g3][g4] = getfill(g4);
						}
					}
					if (dim == 4) {
						ra[g][g1][g2][g3] = getfill(g3);
					}
				}
				if (dim == 3) {
					ra[g][g1][g2] = getfill(g2);
				}
			}
			if (dim == 2) {
				ra[g][g1] = getfill(g1);
			}
		}
		if (dim == 1) {
			ra[g] = getfill(g);
		}
	}//initializes arrays
	
	function getfill(index) {
		if (ifrand == true) {
			if (ifroundrand == true) {
				return rr(randl,randh+1);
			}
			else {
				return random(randl,randh);
			}
		}
		else if (ascending == true) {
			return index;
		}
		else if (typeof fill === 'function') {
			return fill();
		}
		else {
			return fill;
		}
	}
	
	return ra;
	
}//limit of 6 dimensions, randl = lower bound, randh = upper bound

function shapenet(shapeARR,specific,dim,sizing,fill,ifrand,randl,randh,ifroundrand,ascending) {
	
	let totalshape = [];
	if (specific == false) {
		totalshape[0] = [shapeARR[0],shapeARR[1]];
		for (gsn = 1; gsn < sizing; gsn++) {
			totalshape[gsn] = [shapeARR[1],shapeARR[1]];
		}
		totalshape[sizing] = [shapeARR[1],shapeARR[2]];
		if (sizing == 0) {
			totalshape[0] = [shapeARR[0],shapeARR[2]];
		}
	}
	else {
		totalshape = shapeARR;
	}
	let rasn = [];
	if (dim == 1) {
		for (gsn = 0; gsn < totalshape.length; gsn++) {
			totalshape[gsn] = [totalshape[gsn][1]];
		}
	}//change for bias arrays
	for (gsn = 0; gsn < totalshape.length; gsn++) {
		rasn[gsn] = maketensor(dim,[totalshape[gsn][0],totalshape[gsn][1]],fill,ifrand,randl,randh,ifroundrand,ascending);
	}
	return rasn;
	
}

function add2d(ARR1,ARR2) {

	let ra = ARR1;
	for (g = 0; g < ARR1.length; g++) {
		for (g1 = 0; g1 < ARR1[0].length; g1++) {
			ra[g][g1] += ARR2[g][g1];
		}
	}
	return ra;
	
}//adds two same size arrays

function sub2d(ARR1,ARR2) {
	
	let ra = ARR1;
	for (g = 0; g < ARR1.length; g++) {
		for (g1 = 0; g1 < ARR1[0].length; g1++) {
			ra[g][g1] -= ARR2[g][g1];
		}
	}
	return ra;
	
}//adds two same size arrays

function mult2d(ARR1,ARR2) {

	let ra = ARR1;
	for (g = 0; g < ARR1.length; g++) {
		for (g1 = 0; g1 < ARR1[0].length; g1++) {
			ra[g][g1] *= ARR2[g][g1];
		}
	}
	return ra;
	
}//adds two same size arrays

function div2d(ARR1,ARR2) {
	
	let ra = ARR1;
	for (g = 0; g < ARR1.length; g++) {
		for (g1 = 0; g1 < ARR1[0].length; g1++) {
			ra[g][g1] /= ARR2[g][g1];
		}
	}
	return ra;
	
}//adds two same size arrays

function concatenate(ARR) {
	
	let ra = [];
	for (g = 0; g < ARR.length; g++) {
		for (g1 = 0; g1 < ARR[g].length; g1++) {
			ra[ra.length] = ARR[g][g1];
		}
	}
	return ra;
	
}//combines rows of 2d array into 1d array

function normalize(ARR,scalar) {
	
	let arrneg = mult2d([CA(ARR)],[maketensor(1,[ARR.length],-1)])[0];
	let nv = max(max(ARR),max(arrneg))/scalar;//find maximum value
	return div2d([CA(ARR)],[maketensor(1,[ARR.length],nv)])[0];
	
}

function rr(low,top) {
	
	if (low == top) {
		print("ur stupid","go fix ur code -xoxo, rr (low==top error)")
		return top;
	}
	return round(random(low-0.5,top-0.5));
	
}

function getrandin(ARR,low,top,checksum) {
	
	let counter = 0;
	for (g = low; g < top; g++) {
		if (ARR[g] === checksum) {
			counter++;
		}
	}//how many checksums are there
	let randranged = rr(1,top-counter-low+1);//get a random number
	for (g = low; g < top; g++) {
		if (ARR[g] !== checksum) {
			randranged--;
		}
		if (randranged == 0) {
			return g;
		}
	}//number in between low and randr
	
}

function CA(ARR,obj) {
	
	return ARR.slice(0);
	
}//prevents fucking awful javascript auto-pointers (copy array)

function GELU(num) {
	
	return scale*num/(1+pow(e,-1.702*num));
	
}

function sigmoid(num) {
	
	return (1/(1+(pow(e,-1*scale*num))))
	
}

function RELU(num) {
	
	return max(0,scale*num);
	
}
