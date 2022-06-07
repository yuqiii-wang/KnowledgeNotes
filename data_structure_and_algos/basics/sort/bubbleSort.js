function bubbleSort(arr){
    for (var i = 0; i < arr.length - 1; i++){
        for (var j = 0; j < arr.length - 1 - i; j++){
            if (arr[j] < arr[j+1]){
                let tmp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = tmp;
            }
        }
    }
    return arr;
}

arr = [5,4,3,9,8,7,1,2,3]
console.log(bubbleSort(arr));