function runGen(input,sorted,iid,outputarr,allnetworks) {
	
	for (gg = 0; gg < inputsize; gg++) {
		if (activatein == true) {
			allnetworks[iid][0][gg].value = activate([input[gg]])[0];
		}
		else {
			allnetworks[iid][0][gg].value = input[gg];
		}
	}//prepare input
	for (gg = 1; gg < allnetworks[iid].length; gg++) {//layers
		for (gg1 = 0; gg1 < allnetworks[iid][gg].length; gg1++) {//neurons in layer
			allnetworks[iid][gg][gg1].value = 0;
		}
	}//zero out rest of network
	for (gg = 1; gg < allnetworks[iid].length; gg++) {//layers
		for (gg1 = 0; gg1 < allnetworks[iid][gg].length; gg1++) {//neurons in layer
			for (gg2 = 0; gg2 < allnetworks[iid][gg][gg1].weights.length; gg2++) {//layers of weights
				if (allnetworks[iid][gg][gg1].weights[gg2] === undefined) {
					continue;
				}
				for (gg3 = 0; gg3 < allnetworks[iid][gg][gg1].weights[gg2].length; gg3++) {//neuron in layer of weights
					if (allnetworks[iid][gg][gg1].weights[gg2][gg3] === undefined) {
						continue;
					}
					allnetworks[iid][gg][gg1].value += 
						allnetworks[iid][gg][gg1].weights[gg2][gg3]*
						allnetworks[iid][gg2][gg3].value;
				}
			}
			if (addbias == true) {
				allnetworks[iid][gg][gg1].value += allnetworks[iid][gg][gg1].bias;
			}
			allnetworks[iid][gg][gg1].value = activate([neuronstore[iid][gg][gg1].value])[0];
		}
	}//apply weights and biases
	
	let returnarr = [];
	for (gg = 0; gg < outputsize; gg++) {
		returnarr[gg] = allnetworks[iid][allnetworks[iid].length-1][gg].value;
	}
	if (sorted == true) {
		return Bsort(returnarr,outputarr,false);
	}
	else {
		return returnarr;
	}
	
}

function runGPT(last) {

	//multihead self attention
	let adjvalue = CA(encoders[last[last.length-1]]);
	neuronstore = maketensor(3,[layers],0);//init
	for (ll = 0; ll < layers; ll++) {
		let flowingvalues = [];//initialize
		
		//multihead self attention
		for (hh = 0; hh < heads; hh++) {
			let normtensor = maketensor(1,[learningset],0);//init
			let qdot = matrixmult([CA(encoders[last[last.length-1]])],CA(query[ll][hh]));//gets query vec with matrix
			for (b = 0; b < learningset; b++) {
				let encoded = add2d([CA(encoders[last[b]])],[maketensor(1,[encodesize],positioners(b+1))]);//encode position into word vec
				let kdot = matrixmult(encoded,CA(key[ll][hh]));//gets corresponding key vec with matrix
				normtensor[b] = matrixmult(qdot,transpose(kdot))[0][0]/sqrt(querykeydim);//find vec association and scale (basically dot product)
			}//multiply one query with all input based keys
			normtensor = softmax(normtensor);//softmax vector associations
			for (a = 0; a < learningset; a++) {
				//get vector shift with scaled value vec
				let curval = matrixmult([CA(encoders[last[a]])],valuedown[ll][hh]);//use value matrix to get adjustment (down: [qkdim])
				curval = matrixmult(curval,valueup[ll][hh]);//use value matrix to get adjustment (back up to [encode])
				flowingvalues[hh] = mult2d(curval,[maketensor(1,[encodesize],normtensor[a])])[0];//scale by normtensor and add to flow
			}//update desired changes array
		}
		flowingvalues = concatenate(flowingvalues);
		
		//feed forward network
		let linout = runlinear(flowingvalues,ffnlayers,false,weights[ll],biases[ll]);//gets neuron arrangement[1] and new output flow values[0]
		adjvalue = normalize(add2d([adjvalue],[linout[0]])[0],wi);//add and normalize
		neuronstore[ll] = linout[1];//storage for later training
	}//multilayer gpt oh yeah
	
	let finaldot = matrixmult([adjvalue],transpose(encoders))[0];//[0] is just to lower dimension
	return finaldot;
	
}//takes in previous tokens as numbers

function runlinear(input,qlayers,sorted,allweights,allbiases) {
	
	function linear(ARR,weightsarr,biasesarr) {

		let returnarr = matrixmult([ARR],weightsarr)[0];
		if (addbias == true) {
			returnarr = add2d([returnarr],[biasesarr])[0];
		}

		return returnarr;

	}//takes in 1d array and returns one transform with weights from layer + bias

	let nsra = [];
	nsra[0] = input;
	let ra = input;
	for (a = 0; a < qlayers; a++) {
		nsra[a+1] = linear(ra,allweights[a],allbiases[a]);
		ra = activate(nsra[a+1])
	}
	if (sorted == true) {
		return [Bsort(ra,outputarr),nsra];
	}
	else {
		return [ra,nsra];
	}

}//takes in parameters for layers (corr to weights), if sorted, input 
//returns an array of [output , each layer unactivated arr]
