// https://www.geeksforgeeks.org/pyramid-form-increasing-decreasing-consecutive-array-using-reduce-operations/

// there should be only four probabilities: starting or ending with zero or one.
// time complexity should O(n)

function range(start, end) {
    var arr: any;
    if (start < end){
        arr = new Array(end - start + 1);
        arr = arr.fill(undefined).map((_, i) => i + start);
    } else {
        arr = new Array(start - end + 1);
        arr = arr.fill(undefined).map((_, i) => start - i);
    }
    return arr;
}

function makePyramid(arr: Array<number>){
    let mid: number = Math.floor(arr.length / 2);

    let L1: Array<number> = range(0, mid-1);
    let R1: Array<number> = range(mid, 1);
    let newArr1: Array<number> = L1.concat(R1);

    let L2: Array<number> = range(1, mid-1);
    let R2: Array<number> = range(mid, 1);
    let newArr2: Array<number> = L2.concat(R2);

    let L3: Array<number> = range(1, mid-1);
    let R3: Array<number> = range(mid, 0);
    let newArr3: Array<number> = L3.concat(R3);

    let L4: Array<number> = range(0, mid-1);
    let R4: Array<number> = range(mid, 0);
    let newArr4: Array<number> = L4.concat(R4);

    let L5: Array<number> = range(0, mid);
    let R5: Array<number> = range(mid+1, 1);
    let newArr5: Array<number> = L5.concat(R5);

    let L6: Array<number> = range(1, mid);
    let R6: Array<number> = range(mid+1, 1);
    let newArr6: Array<number> = L6.concat(R6);

    let L7: Array<number> = range(1, mid);
    let R7: Array<number> = range(mid+1, 0);
    let newArr7: Array<number> = L7.concat(R7);

    let L8: Array<number> = range(0, mid);
    let R8: Array<number> = range(mid+1, 0);
    let newArr8: Array<number> = L8.concat(R8);

    let bigArr: Array<Array<number>> = [newArr1, newArr2, newArr3, newArr4,
                                        newArr5, newArr6, newArr7, newArr8];

    let minReduce: number = Infinity;
    for (let i = 0; i < 8; i++){
        let tmpReduce:  number = 0;
        if (bigArr[i].length == arr.length){
            for (let j = 0; j < arr.length; j++){
                tmpReduce += arr[j] - bigArr[i][j];
            }
            if (tmpReduce < minReduce && tmpReduce >= 0)
                minReduce = tmpReduce;
        }
    }
    if (minReduce == Infinity)
        return 0;
    return minReduce;
}

console.log(makePyramid([1,2,3,4,3,1]));
console.log(makePyramid([1,2,1]));
console.log(makePyramid([1,2,1,1]));
console.log(makePyramid([1,3,2,1,1]));
console.log(makePyramid([1]));