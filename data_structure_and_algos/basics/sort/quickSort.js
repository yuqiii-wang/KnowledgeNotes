function swap(arr, idx_1, idx_2){
    let tmp = arr[idx_1];
    arr[idx_1] = arr[idx_2];
    arr[idx_2] = tmp;
}

function partition(arr, leftIdx, rightIdx){
    let pivot = arr[rightIdx];
    let i = leftIdx;
    for(iter = leftIdx; iter < rightIdx; ++iter){
        if (arr[iter] <= pivot)
            swap(arr, iter, i++);
    }
    swap(arr, rightIdx, i);
    return i;
}

function quickSort(arr, leftIdx, rightIdx){
    if (leftIdx < rightIdx){
        let newIdx;
        newIdx = partition(arr, leftIdx, rightIdx);
        quickSort(arr, leftIdx, newIdx - 1);
        quickSort(arr, newIdx + 1, rightIdx);
    }
}

arr = [4,3,2,5,6,7,1,9,8,9];
quickSort(arr, 0, arr.length - 1);
console.log(arr);