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
	
}

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

function normalize(ARR,scalar) {
	
	let arrneg = mult2d([CA(ARR)],[maketensor(1,[ARR.length],-1)])[0];
	let nv = max(max(ARR),max(arrneg))/scalar;//find maximum value
	return div2d([CA(ARR)],[maketensor(1,[ARR.length],nv)])[0];
	
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
