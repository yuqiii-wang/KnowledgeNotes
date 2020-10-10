function mergeSort(arr: Array<number>){
    if (arr.length > 1) {
        let mid: number = Math.ceil(arr.length / 2);
        let L : Array<number> = arr.slice(0, mid);
        let R : Array<number> = arr.slice(mid, arr.length);
        mergeSort(L);
        mergeSort(R);

        let idx_l: number = 0; let idx_r: number = 0; let idx_k: number = 0;
        while (idx_l < L.length && idx_r < R.length) {
            if (L[idx_l] < R[idx_r]){
                arr[idx_k] = L[idx_l];
                ++idx_l;
            } else {
                arr[idx_k] = R[idx_r];
                ++idx_r;                
            }
            ++idx_k;
        }
        while (idx_l < L.length) {
            arr[idx_k] = L[idx_l];
            ++idx_k;
            ++idx_l;
        }
        while (idx_r < R.length) {
            arr[idx_k] = R[idx_r];
            ++idx_k;
            ++idx_r;
        }
    }
}
var arr: Array<number> = [5,6,2,1,4,3,7,8];
mergeSort(arr);
console.log(arr);