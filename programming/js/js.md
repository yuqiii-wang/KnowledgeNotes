# JavaScript

## DOM Element Manipulation

JavaScript is designed to manipulate DOM (Document Object Model) elements to control UI display.

DOM is used to render a UI page, where compiler, e.g., chrome V8 engine, parses a DOM to a tree-structured elements and display such elements per style.
User can interact with DOM elements.

For example, this DOM

```html
<!DOCTYPE html>
<html>
<head>
    <title>My Web Page</title>
</head>
<body>
    <h1>Welcome to My Web Page</h1>
    <p>This is a simple paragraph.</p>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
        <li>Item 3</li>
    </ul>
</body>
</html>
```

is parsed to

```txt
Document
├── html
│   ├── head
│   │   └── title ("My Web Page")
│   └── body
│       ├── h1 ("Welcome to My Web Page")
│       ├── p ("This is a simple paragraph.")
│       └── ul
│           ├── li ("Item 1")
│           ├── li ("Item 2")
│           └── li ("Item 3")
```

For any element change, traditionally, the whole UI page needs re-rendering.
React is used to prevent full webpage change by taking DOM partial update only.

React relies on a virtual DOM, which is a copy of the actual DOM.
React's virtual DOM is immediately reloaded to reflect this new change whenever there is a change in the data state.
After which, React compares the virtual DOM to the actual DOM to figure out what exactly has changed.

### Common DOM Elements

|Element|Display Type|Primary Use Case & Key Differences|
|:---|:---|:---|
|`<div>`|Block|Generic use for block|
|`<span>`|Inline|Used to hook onto a piece of text for styling (e.g., changing color, adding an icon) without affecting layout.|
|`<p>`|Block|For a paragraph of text. It is not for grouping other elements.|
|`<ul>`, `<ol>`|Block|Containers for unordered (bulleted) and ordered (numbered) lists, respectively. Their only valid direct child is the `<li>` element.|
|`<li>`|List Item|It must be a child of a `<ul>` or `<ol>`. Behaves mostly like a block element but has a list marker (bullet/number).|

where for `display: block;` vs `display: inline;`, the `inline` display aims to fit content within only necessary space, e.g., ignores width and height.

## Single thread and Event Loop

* JS is single-thread

JS is an asynchronous and single-threaded interpreted language. A single-thread language is one with a single call stack and a single memory heap.

* Event Loop

Call Stack: If your statement is asynchronous, such as `setTimeout()`, `ajax()`, `promise`, or click event, the event is pushed to a queue awaiting execution.

Queue, message queue, and event queue are referring to the same construct (event loop queue). This construct has the callbacks which are fired in the event loop.

## Inheritance by Prototype Chain

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

## JS, TS and TSX Grammar

|Feature|`.js`|`.ts`|`.tsx`|
|:---|:---|:---|:---|
|Language|JavaScript|TypeScript|TypeScript|
|JSX Support|Yes (with a tool like Babel)|No|Yes|
|Type Safety|No|Yes|Yes|
|Common Use|Basic React components|Non-component TypeScript logic|React components in TypeScript|

### Basic Data Types

For JavaScript:

```js
// 1. String: For text
let employeeName = "John Doe";
console.log(typeof employeeName); // "string"

// 2. Number: For both integers and decimals
let salary = 95000.50;
console.log(typeof salary); // "number"

// 3. Boolean: For true/false values
let isActive = true;
console.log(typeof isActive); // "boolean"

// 4. Undefined: A variable that is declared but not assigned a value
let manager;
console.log(manager); // undefined
console.log(typeof manager); // "undefined"

// 5. Null: Intentionally represents "no value"
let directReport = null;
console.log(directReport); // null
console.log(typeof directReport); // "object" (This is a famous quirk in JavaScript!)

// 6. BigInt: For integers larger than the max safe number
const veryLargeNumber = 900719925474099199n; // The 'n' suffix is important
console.log(typeof veryLargeNumber); // "bigint"

// 7. Symbol: To create unique identifiers
const idSymbol = Symbol('uniqueId');
console.log(typeof idSymbol); // "symbol"
```

For TypeScript:

```ts
// 1. : string
let employeeName: string = "Alice";

// 2. : number
let salary: number = 1200.50;

// 3. : boolean
let isActive: boolean = true;

// 4. : null
// To explicitly allow a variable to be null, you often use a union type.
// This is more common than just `: null`.
let supervisor: string | null = "Bob";
supervisor = null; // Valid

// 5. : undefined
// A variable annotated to only be `undefined`.
let bonus: undefined = undefined;

// 6. : bigint
const veryLargeNumber: bigint = 900719925474099199n;
// veryLargeNumber = 123; // Error: Type 'number' is not assignable to type 'bigint'.

// 7. : symbol
const uniqueId: symbol = Symbol("id");
```

#### `null` vs `undefined`

|Feature|undefined|null|
|:---|:---|:---|
|Meaning|A variable has not been assigned a value.|The intentional absence of a value.|
|Assignment|Usually set automatically by the JavaScript engine.|Assigned deliberately by the user.|
|Use Case|Indicates an uninitialized or missing state.|Indicates a cleared or empty state.|
|`typeof` operator|`typeof undefined` returns `"undefined"`.|`typeof null` returns `"object"`.|
---

For example, if to force to retrieve a return value from a non-return function, the returned value is assigned `undefined` by compiler. 

```ts
function logMessage(message) {
  console.log(message);
  // No "return" statement, so the function implicitly returns undefined
}
let result = logMessage("Hello");
console.log(result); // Output: undefined
```

While for `null`, it is usually assigned by user with semantics.

```ts
let selectedUser = { name: "Alice" };
// ... later, the user clicks "clear selection"
selectedUser = null; // Deliberately setting it to have no value
```

### Assignment: Value-Copy vs Reference

#### Primitives are Value-Copied

Assigning Primitive Types (Copy by Value)
Primitives are simple, immutable data types: `string`, `number`, `boolean`, `null`, `undefined`, `symbol`, `bigint`.

```js
let a: number = 10;
let b: number = a; // The value 10 is COPIED into b

console.log(a); // Output: 10
console.log(b); // Output: 10

// Now, let's change b
b = 20;

console.log(a); // Output: 10 (a is completely unaffected)
console.log(b); // Output: 20
```

#### Reference Types

Reference types are complex data structures like objects and arrays.

```js
// An object is a reference type
let postA = {
  title: "My First Post",
  likes: 100
};

// This does NOT create a new object.
// It copies the MEMORY ADDRESS of postA into postB.
let postB = postA;

console.log(postA.likes); // Output: 100
console.log(postB.likes); // Output: 100

// Now, let's modify the object using postB
postB.likes = 150;

// The change is reflected in BOTH variables because they point to the SAME object.
console.log(postA.likes); // Output: 150 (postA was also changed!)
console.log(postB.likes); // Output: 150
```

### Variable Scope: `var` vs `let`

Variables declared with `var` have function scope, while `let` variables are of block scope.

`var i` outlives the `for (var i = 0; i < 3; i++) {...}` scope that the `i` is accessible after the `for` loop has finished.

```js
function testVarScope() {
  // `i` is accessible here, even before the loop, but its value is `undefined`.
  console.log("Before loop:", i); // Output: undefined

  for (var i = 0; i < 3; i++) {
    console.log("Inside loop:", i); // Output: 0, 1, 2
  }

  // `i` has "leaked" out of the loop and is still accessible here.
  console.log("After loop:", i); // Output: 3
}
```

Once the `for` loop has finished, the block scope of `i` is destroyed.

```js
function testLetScope() {
  // console.log(i); // This would cause a ReferenceError. `i` is not defined here.

  for (let i = 0; i < 3; i++) {
    console.log("Inside loop:", i); // Output: 0, 1, 2
  }

  // console.log(i); // This causes a ReferenceError because `i` does not exist here.
}
```

### Some Novel JS Syntax

#### Expansion `...`

It has two use scenarios:

* Collection Merge

```js
const firstHalf: number[] = [1, 2, 3];
const secondHalf: number[] = [4, 5, 6];

// Using spread to combine them
const combined: number[] = [...firstHalf, ...secondHalf];
console.log(combined); // Output: [1, 2, 3, 4, 5, 6]
```

* Collection Reset

```js
// The '...numbers' syntax gathers all arguments passed to the function into an array called 'numbers'.
// TypeScript requires you to type the array, e.g., number[].
function sumAll(...numbers: number[]): number {
  return numbers.reduce((total, current) => total + current, 0);
}

console.log(sumAll(1, 2));          // Output: 3
console.log(sumAll(10, 20, 30));    // Output: 60
console.log(sumAll(5));             // Output: 5
console.log(sumAll());              // Output: 0
```

### Some Novel TS Syntax

#### Interface

```ts
interface Person {
  name: string;
  age: number;
  isStudent?: boolean; // Optional property
}

function greet(person: Person) {
  return `Hello, ${person.name}!`;
}
```

#### Optional Chaining Operator: `?.`

`?.` returns `undefined` if element not existed, instead of throwing an error.

```ts
const users = [
  { name: "Alice" },
  { name: "Bob" }
];

// Safely access the name of the first user
const firstName = users?.[0]?.name; // "Alice"

// Safely attempt to access an element that doesn't exist
const nonExistentName = users?.[5]?.name; // undefined
```

#### The Nullish Coalescing Operator (`??`)

`??` returns the right-hand side operand if the left-hand side operand is `null` or `undefined`.
Otherwise, it returns the left-hand side operand.

The `||` operator treats any "falsy" value (like `0`, an empty string `""`, or `false`) as a trigger to use the default value. The `??` operator is more precise, only falling back to the default for `null` or `undefined`.


```ts
let volume = localStorage.getItem('volume'); // This could be "0"

let a = volume || 0.5; // If volume is "0", `a` becomes 0.5 (undesirable)
let b = volume ?? 0.5; // If volume is "0", `b` remains "0"
```

## Compiler: Node JS

* npx vs npm

NPM is a package manager used to install, delete, and update Javascript packages on your machine.
NPX is a package executer, and it is used to execute javascript packages directly, without installing them.

For example, `npx`

## ES5 vs ES6

## Miscellaneous JS Knowledge

* canvas vs svg
