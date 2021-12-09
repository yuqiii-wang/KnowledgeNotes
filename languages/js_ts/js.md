#  JavaScript Grammar

* `var` vs `let`

`var` is function scoped and `let` is block scoped.

* `async` and `await`

`await` should be used inside `async`
```js
async function task(){
    return 1;
}
async function run() {
    // Your async code here
    const exampleResp = await task();
}
run();
```