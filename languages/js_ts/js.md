#  JavaScript

## Intro

* JS is single-thread

JS is an asynchronous and single-threaded interpreted language. A single-thread language is one with a single call stack and a single memory heap.

* Event Loop

Call Stack: If your statement is asynchronous, such as setTimeout(), ajax(), promise, or click event, the event is pushed to a queue awaiting execution.

Queue, message queue, and event queue are referring to the same construct (event loop queue). This construct has the callbacks which are fired in the event loop.

* Prototype

The prototype is an object that is associated with every functions and objects by default in JavaScript.

Example:
`studObj1.age = 15;` is not broadcast to `studObj2` in the code given below.
```js
function Student() {
    this.name = 'John';
    this.gender = 'Male';
}

var studObj1 = new Student();
studObj1.age = 15;
alert(studObj1.age); // 15

var studObj2 = new Student();
alert(studObj2.age); // undefined
```

Provided the nature of JS with prototype implementation, `age` attribute can be shared across all derived objects of `Student`. 
```js
Student.prototype.age = 15;
```

## Grammar

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