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
				return rr(randl,randh);
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
	
	let totalshape;
	if (specific == false) {
		totalshape = [shapeARR[0]];
		for (gsn = 1; gsn < sizing+1; gsn++) {
			totalshape[gsn] = shapeARR[1];
		}
		totalshape[sizing+1] = shapeARR[2];
	}
	else {
		totalshape = shapeARR;
	}
	let rasn = [];
	if (dim == 1) {
		totalshape = transpose([totalshape])
	}
	for (gsn = 0; gsn < totalshape.length; gsn++) {
		rasn[gsn] = maketensor(dim,[totalshape[gsn][0],totalshape[gsn][1]],fill,ifrand,randl,randh,ifroundrand,ascending);
	}
	return rasn;
	
}
