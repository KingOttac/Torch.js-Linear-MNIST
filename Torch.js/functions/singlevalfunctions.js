function GELU(num) {
	
	return scale*num/(1+pow(e,-1.702*num));
	
}

function sigmoid(num) {
	
	return (1/(1+(pow(e,-1*scale*num))))
	
}

function RELU(num) {
	
	return max(0,scale*num);
	
}

function rr(low,top) {
	
	if (low == top) {
		print("ur stupid","go fix ur code -xoxo, rr (low==top error)")
		return top;
	}
	return round(random(low-0.5,top-0.5));
	
}

function getfuncderiv(input) {
		
	if (type == "sigmoid") {
		return (scale*pow(e,-1*scale*input))/pow(1+pow(e,-1*scale*input),2);
	}
	else if (type == "RELU") {
		return 1;
	}
	else if (type == "GELU") {
		let ndx = pow(e,1.702*input);
		return (ndx*scale*(1+ndx+1.702*input))/pow(1+ndx,2)
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

function adamW(parr,p0,p1,p2,p3,pv) {

	let cc = correctcheck[pv][p0][p1][p2][p3];
	if (abs(cc[cc.length-2]-cc[cc.length-1]) >= convthresh) {
		//move in from array
		let mtin = mt[pv][p0][p1][p2][p3];
		let vtin = vt[pv][p0][p1][p2][p3];
		let tspin = tsp[pv][p0][p1][p2][p3];

		//calculate vec adjust
		tspin++;
		let randomspot = rr(learningset,convertedlines.length);
		let rereturn = runexample(randomspot);
		let gtin = cc[cc.length-1]-rereturn;
		mtin = b1*mtin + (1-b1)*gtin;//get first vec change
		vtin = b2*vtin + (1-b2)*pow(gtin,2);//get second vec change
		if (pv == 5) {
			parr[p0][p1][p2] -= alpha*(mtin/(1-pow(b1,tspin)))/(sqrt(vtin/(1-pow(b2,tspin))) + epsilon);
		}//bias changes
		else {
			parr[p0][p1][p2][p3] -= alpha*(mtin/(1-pow(b1,tspin)))/(sqrt(vtin/(1-pow(b2,tspin))) + epsilon);
		}//everything else

		//move back changed values
		mt[pv][p0][p1][p2][p3] = mtin;
		vt[pv][p0][p1][p2][p3] = vtin;
		tsp[pv][p0][p1][p2][p3] = tspin;
		correctcheck[pv][p0][p1][p2][p3][cc.length] = rereturn;
	}//vector adjustment

}//implements adamW optimizer
