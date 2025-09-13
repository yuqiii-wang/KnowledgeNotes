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

Modern JS uses features like `Promises` and the `async`/`await` syntax to give better performance.

### Parallelism by Web Workers

Web Workers can give multiple threads for execution.

## JS Inheritance by Prototype

JS is a scripting language not required compilation such as observed in c++ and java, as a result, the JS hidden property `Prototype` much more flexible in inheritance against c++ or java parent class.

For example, `Person.prototype.greet` can be added just in time without prior declaration that is often observed in c++ and java.

```js
// 1. Define a constructor function
function Person(name, age) {
  this.name = name;
  this.age = age;
}

// 2. Add a method to the Person function's prototype object
Person.prototype.greet = function() {
  console.log(`Hello, my name is ${this.name}.`);
};

// 3. Let's inspect the prototype itself
console.log(Person.prototype);

// greet: f(): The function we just defined.
// constructor: f Person(name, age): A property that points back to the Person function itself. This is a default property on every function's prototype.
// __proto__: Object: This shows that the Person.prototype object itself has a prototype, which is the base Object.prototype. This is the foundation of the prototype chain.

// 4. Create an instance of Person
const person1 = new Person('Alice', 30);

// 5. Inspect the instance
console.log(person1);

// name: "Alice"
// age: 30
// __proto__: Object: This is the crucial link. It points to the object that serves as the prototype for person1.
```

Now create inheritance `Student`.

```javascript
function Student(name, age, studentId) {
  Person.call(this, name, age);
  this.studentId = studentId;
}

Student.prototype = Object.create(Person.prototype);
Student.prototype.constructor = Student;

// Add a method specific to Student
Student.prototype.study = function() {
  console.log(`${this.name} is studying.`);
};

const student1 = new Student('Bob', 22, 'S12345');

student1.greet(); // Output: Hello, my name is Bob. (Inherited!)
student1.study();    // Output: Bob is studying. (Own prototype method)

console.log('\n--- Inspecting the prototype chain ---');
console.log('Student instance:', student1);
console.log('Student.prototype:', Student.prototype);
console.log('Person.prototype:', Person.prototype);
```

### Prototype Chain

To access a property on an object, JavaScript's engine will

1. first look for the property on the object itself
2. if it doesn't find it, it will then look at the Object's prototype
3. continues up what is known as the prototype chain until `Object`
4. if still not found the end of the chain is reached (which is `null`)

```js
// This demonstrates the chain:
// student1 -> Student.prototype -> Person.prototype -> Object.prototype -> null
console.log(Object.getPrototypeOf(student1) === Student.prototype); // true
console.log(Object.getPrototypeOf(Student.prototype) === Person.prototype); // true
```

### Common Confusion: `__proto__` vs `prototype`

* `__proto__` serves as pointer for fast object location
* `prototype` is the actual object

To execute `student1.study();`

1. JS checks `student1` for a study method. Not found.
2. JS follows `student1.__proto__` to `Student.prototype`.
3. JS checks `Student.prototype` for a study method. Found! It executes the function.

To execute `student1.greet();`

1. JS checks `student1` for a greet method. Not found.
2. JS follows `student1.__proto__` to `Student.prototype`.
3. JS checks `Student.prototype` for a greet method. Not found.
4. JS follows `Student.prototype.__proto__` to `Person.prototype`.
5. JS checks `Person.prototype` for a greet method. Found! It executes the function.

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

## Execution Env: Node JS vs Browser Chrome

* Browser: The browser provides a **client-side execution environment**

Designed to create interactive and dynamic web pages, manipulate the Document Object Model (DOM), and respond to user events like clicks and keyboard presses, and communicate with servers.

JavaScript runs in a browser-restricted sandboxed environment with limited access to computer resources, e.g., filesystem, OS executables.

* Node.js: Node.js offers a **server-side runtime environment**

Allow developers to use JavaScript to build back-end services, APIs, command-line tools, and other applications that run outside of a browser.

Node.js does not have a DOM because it doesn't render HTML pages.
Consequently, user cannot access objects like `document` or `window` in a Node.js environment.

Node.js has full access to the system's resources.

### Node JS

* npx vs npm

NPM is a package manager used to install, delete, and update Javascript packages on your machine.
NPX is a package executer, and it is used to execute javascript packages directly, without installing them.

For example, `npx`

## ES5 vs ES6

## Miscellaneous JS Knowledge

* canvas vs svg
