# Typical React Interview Questions

## React State vs Property

||Attribute|Props|State|
|:---|:---|:---|:---|
|Purposes|Defines HTML element characteristics. In JSX, they become props.|Passes data from parent to child.|Manages a component's internal, changing data.|
___

* State

State is local and private to the component where it is defined.

The `count` is state of `Counter`.
When `setCount` is invoked that triggers state update, the `Counter` will be re-rendered.

```js
import React, { useState } from 'react';

function Counter() {
  // Declare a state variable named "count"
  // setCount is the function to update it
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>You clicked {count} times</p>
      <button onClick={() => setCount(count + 1)}>
        Click me
      </button>
    </div>
  );
}
```

Regular JS variables are NOT states, React does NOT track the change of such variables, hence UI will not update accordingly.

```js
function MyComponent() {
  let count = 0; // This is NOT state.

  const handleClick = () => {
    count = count + 1; // This change will NOT re-render the component.
    console.log(count); // The console will show the new value, but the UI won't update.
  };

  return <button onClick={handleClick}>Count is {count}</button>;
}
```

* Props

Props are read-only and immutable within the receiving component.

Props are used to pass data or functions down to child components.

For example, on `onClick={incrementCount}`, the state `count` is updated.
`CounterDisplay` is a child component of `CounterApp` that receives `count` as a prop as an argument to its view.

```js
import React, { useState } from 'react';

// Child component
const CounterDisplay = ({ count }) => {
  return <h2>Current Count: {count}</h2>;
};

// Parent component
const CounterApp = () => {
  // State to keep track of the count
  const [count, setCount] = useState(0);

  // Function to increment the count
  const incrementCount = () => {
    setCount(count + 1);
  };

  return (
    <div>
      <CounterDisplay count={count} /> {/* Passing state as props */}
      <button onClick={incrementCount}>Increment Count</button>
    </div>
  );
};

export default CounterApp;
```

* Attribute

Attribute refers to traditional/default props, e.g., `src` of the DOM `img`.

```js
<img id="main-logo" class="logo" src="/logo.png" alt="Company Logo">
```

### On What Conditions React Renders

By default, a component will re-render if **its parent re-renders**, or if **its own state or props change**.

## React Component Lifecycle

1. **Mounting Phase**

1.1 `constructor` Called before the component is mounted.

```js
constructor(props) {
  super(props);
  this.state = { count: 0 };
}
```

1.2 `render()`

Required method that returns JSX to define the UI.

1.3 `componentDidMount()`

Invoked after the component is added to the DOM.

2. **Updating Phase**

2.1 `shouldComponentUpdate(nextProps, nextState)`

Determines if the component should re-render.
Returning `false` skips rendering and improves performance.

```js
shouldComponentUpdate(nextProps, nextState) {
  return nextState.count !== this.state.count;
}
```

2.2 `render()`

Called to update the UI when state or props change.

2.3 `componentDidUpdate(prevProps, prevState, snapshot)`

Invoked after the DOM is updated.

3. **Unmounting Phase**

3.1 `componentWillUnmount()`

Used to clean up resources like event listeners, timers, or subscriptions.

### Example: A Counter

1. On mounting

```txt
Constructor called
Render method called
Component did mount
```

2. On Updating (Click Increment Button)

```txt
Should component update
Render method called
Component did update
```

3. On Unmounting

```txt
Component will unmount
```

Code:

```js
import React from 'react';

class Counter extends React.Component {
  constructor(props) {
    super(props);
    this.state = { count: 0 };
    console.log('Constructor called');
  }

  componentDidMount() {
    console.log('Component did mount');
  }

  shouldComponentUpdate(nextProps, nextState) {
    console.log('Should component update');
    return nextState.count !== this.state.count;
  }

  componentDidUpdate(prevProps, prevState) {
    console.log('Component did update');
  }

  componentWillUnmount() {
    console.log('Component will unmount');
  }

  increment = () => {
    this.setState((prevState) => ({ count: prevState.count + 1 }));
  };

  render() {
    console.log('Render method called');
    return (
      <div>
        <h1>Count: {this.state.count}</h1>
        <button onClick={this.increment}>Increment</button>
      </div>
    );
  }
}

export default Counter;
```

### `useEffect` invocation in different lifecycles

`useEffect` serves purposes similar to the lifecycle methods in class components, such as `componentDidMount`, `componentDidUpdate`, and `componentWillUnmount`.

`useEffect` with empty array dependency runs

* once on mount
* multiple times after every render
* once on unmount for cleanup

```js
useEffect(() => {
  ...
}, []); // Empty dependency array ensures it runs once on mount, 
        // multiple times after every render, and once on unmount for cleanup
```

## Inheritance in React/JavaScript (Prototype Chain)

### Prototype Chain

When access a property or method on an object, JavaScript first looks for it on the object itself.
If not found, it looks at the object's prototype, then the prototype's prototype, and so on, until it reaches `Object` and further to `null`.

Inheritance in JavaScript is achieved via the prototype chain to inherit properties and methods from other objects, and React class components inherit from `React.Component`.

Every JavaScript object has an internal property called `[[Prototype]]`.

The code sets `animal` as the prototype of `dog`.

```js
const animal = {
    eat() { console.log("Eating..."); }
};

const dog = {
    bark() { console.log("Barking..."); }
};

Object.setPrototypeOf(dog, animal); // Set animal as the prototype of dog
```

`__proto__` is the reference to its parent prototype.
It shows the inheritance chain:
`dog` -> `animal` -> `Object` -> `null`.

```js
console.log(dog.__proto__ === animal); // true
console.log(animal.__proto__ === Object.prototype); // true
console.log(Object.prototype.__proto__ === null); // true
```

#### Prototype Chain In React

React components are usually created using classes or functions (function components). When using class components, JavaScript's prototype chain comes into play.

React components inherit from React.Component, so they get access to methods like `setState()` and lifecycle methods (e.g., `componentDidMount()`).

### `constructor`, `super` and `props`

* `constructor(props)`: The constructor takes props as an argument to initialize the component and passes it to the parent class (`React.Component`) through super(`props`).
* `super(props)`: This calls the parent class's constructor method (`React.Component`) and allows the current class to inherit the properties and methods from `React.Component`. It also gives access to `this.props` in the component.
* `props`: These are the properties that are passed down from the parent component. They are available inside the constructor, and later in the component, using `this.props`.

### `this` Context

`this` is a keyword that refers to the execution context (the object) in which a function is called.
The value of `this` depends entirely on how the function is invoked.

* Global Context

In non-strict mode, `this` refers to the global object (window in browsers or global in Node.js).

In strict mode, `this` is undefined when not explicitly bound.

* Object Context vs Standalone Function

When a function is called as a method of an object, `this` refers to the object itself.

If a method is extracted and invoked as a standalone function, `this` loses its object context and defaults to the global object (undefined in strict mode).

```js
const obj = {
    name: "John",
    sayHello() {
        console.log(this.name);
    }
};

// Object Context
obj.sayHello(); // Logs "John"

// Standalone Function
const extracted = obj.sayHello;
extracted(); // Logs 'undefined' or error, because `this` refers to global object
```

* Event Handlers

In regular functions used as event handlers, `this` typically refers to the DOM element that triggered the event.

```js
const button = document.querySelector("button");
button.addEventListener("click", function () {
    console.log(this); // Logs the button element
});
```

* Arrow Functions

`this` is lexically bound in arrow functions, meaning it inherits this from the enclosing scope.

```js
const obj = {
    name: "John",
    sayHello: () => {
        console.log(this.name);
    }
};

obj.sayHello(); // Logs 'undefined', because `this` refers to the global object
```

#### `this` and `bind`

The bind method allows you to manually set the value of this for a function, regardless of how or where the function is invoked.

```js
const obj = { name: "Alice" };

function greet() {
    console.log(this.name);
}

const boundGreet = greet.bind(obj); // Bind `this` to `obj`
boundGreet(); // Logs "Alice"
```

## React Rendering Event Process Flow

### How React responds on `onClick`

#### Synthetic Event System

React's `SyntheticEvent` is a wrapper around the native DOM event,
designed to provide a consistent and efficient API across all browsers.

##### Lifecycle of a SyntheticEvent

1. Event Creation:

When a user interaction triggers an event (e.g., click), React captures it via its event delegation system.

A new `SyntheticEvent` is created by copying the relevant properties from the native event.

2. Event Handling:

The SyntheticEvent is passed to the appropriate event handler(s) in React component tree.

3. Event Pooling:

After the event handlers have been invoked, React nullifies the properties of the SyntheticEvent and returns it to the pool for reuse.

##### Common Properties

* `type`: The type of the event (e.g., "click", "mouseover").
* `target`: The DOM element that triggered the event.
* `currentTarget`: The DOM element to which the event handler is attached.

##### Common Methods

* `preventDefault()`: Prevents the default action associated with the event (e.g., navigating a link).
* `isDefaultPrevented()`: Returns true if preventDefault() has been called.

#### Event Registration, Delegation to DOM, and Pooling

##### Event Registration

React does not attach the event listener directly to the DOM element. Instead, it uses event delegation.

React attaches a single event listener to the root of the DOM tree (e.g., document or container).

When the DOM is rendered and hydrated, React maps your `onClick` prop to an internal listener.

##### Event Delegation

React's event delegation mechanism works as follows:

All events are captured by a single event listener at the top level (e.g., `document.addEventListener`).

When an event occurs (e.g., `click`), the top-level listener captures it.

React determines which component the event is for by traversing down the DOM tree using event propagation (bubbling).

##### Event Pooling

React reuses `SyntheticEvent` objects for performance optimization. Here's how:

A `SyntheticEvent` is created when the event is triggered.

The properties of the native event (e.g., `target`, `currentTarget`, `type`) are copied into the `SyntheticEvent` object.

After the event has been processed, React releases it back to the pool, making it available for reuse.

##### Dispatching the Event

React matches the event with the corresponding component's event handler by traversing the virtual DOM tree.

### React rendering `useState` on change

1. When a component renders for the first time, React creates an internal "fiber node" to represent it.
This fiber node tracks all the state and hooks used by the component.
2. React uses a concept called the "hook pointer", which is part of the fiber node.
The hook pointer ensures that `useState` calls are tied to specific positions in the component's render tree.
3. React maintains a `memoizedState` property within the fiber node to store the state values of the component.
4. React creates an update object representing a change (e.g., `setState`), and put the update into a queue `queue.pending`.
5. React marks the fiber node as needing an update.
React schedules re-render of the component concerned of efficiency optimization and priority (e.g., user input is higher priority than background tasks).
6. React applies all the updates by traversing the `queue.pending` linked list.

#### Data Structures: Fiber and Hook

* Fiber

Fiber is a tree-like data structure, and each component represents a fiber node.
Fiber node tracks all the state and hooks used by the component.

* Hook

A hook is a lightweight object stored in the `memoizedState` property of a fiber node.

```js
{
  memoizedState: any,    // The state or value for this hook
  queue: {               // A queue of updates for this hook
    pending: null        // Points to a linked list of updates
  },
  next: Hook | null      // Points to the next hook in the list
}
```

##### Example

Given a Fiber Tree,

```txt
App (Fiber Node)
├── Header (Fiber Node)
├── Content (Fiber Node)
│    ├── memoizedState (Linked List of Hooks)
│    │    ├── Hook 1: { useState value }
│    │    ├── Hook 2: { useEffect cleanup }
│    │    └── Hook 3: { useRef current }
└── Footer (Fiber Node)
```

for a component using `useState` and `useEffect`, a Hooks Linked List is

```txt
Fiber.memoizedState -> Hook 1 -> Hook 2 -> null
```

### React Rendering concerning Promise, and `async`/`await`

A `Promise` in JavaScript represents the eventual completion (or failure) of an asynchronous operation and its resulting value.
`async`/`await` is syntactic sugar over `Promises`.

A `Promise` has three statuses:

* Pending
* Fulfilled
* Rejected

A `Promise` object's constructor takes two arguments `resolve`, `reject`,
and by `then` to get fulfilled result, by `catch` to get rejected result.

```js
const promise = new Promise((resolve, reject) => {
  const success = true; // Simulated condition
  if (success) {
    resolve("Task completed!");
  } else {
    reject("Task failed!");
  }
});

// Handling the result
promise
  .then((result) => console.log(result)) // Called on resolve
  .catch((error) => console.error(error)); // Called on reject
```

#### `Promise` Practical Use Cases

Typically, there are two `then`s and one `catch` followed after a `fetch`.
One `then` for response ok check, and another one `then` for loading data.

```js
useEffect(() => {
    // Fetch data using Promises
    fetch("https://api.example.com/data")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((result) => {
        setData(result); // Update state with the resolved data
        setLoading(false); // Mark loading as false
      })
      .catch((err) => {
        setError(err.message); // Handle error
        setLoading(false);
      });
  }, []);
```

When used with `async`, ensures the effect runs after the component mounts or when dependencies change.
In other words, inside `fetchData()`, the `setData(result);` must be after `const result = await response.json();`.

```js
useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("https://api.example.com/data");
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false); // Ensure loading state is updated
      }
    };
    fetchData();
  }, []);
```

## Virtual DOM vs Actual DOM

* Actual DOM

It represents the structure of an HTML or XML document as a tree of nodes that actually interact with rendering engine, e.g., web browser.

However, directly manipulating the actual DOMs is heavy.
There should be a lightweight preceding operation that optimizes todo actual DOMs.

* Virtual DOM

Virtual DOM is a lightweight, in-memory representation of the actual DOM as a JavaScript object and is not directly tied to the browser's rendering engine.

### How React Uses the Virtual DOM: The Reconciliation Process

1. When the state of a component changes, React creates a new Virtual DOM tree that reflects this updated state.
2. React then compares this new Virtual DOM tree with the previous one. This comparison process is known as "diffing."
3. React calculates the most efficient way to apply these changes to the real DOM with batched pending updates.
4. React renders the actual changes to actual DOM.

### Diffing and `key`

React uses diffing algo with `key` to determine if a DOM element needs to get created/destroyed.

Despite that the `key` for each row remains the same, the key prop's purpose is **for reconciliation and identification, not for preventing re-renders**.
As a result, if any of item in `initialUsers` got changed by `setUsers`, all four users are updated.
This is triggered by the state `users` of `App` updated as a parent, whose children the four `UserRow` are updated by a cascading effect.

React uses the key to efficiently determine:

* Which item was added?
* Which item was removed?
* Which item was re-ordered?

By having a stable identity (`key`), React can avoid destroying and re-creating DOM nodes unnecessarily and can preserve the state of components within the list.
In other words, whether DOM nodes are to re-rendered does not concern the `key`, and diffing is about DOM node construction/destruction, not DOM's prop.

```js
import React, { Component } from 'react';

const initialUsers = [
  { id: 1, name: 'Alice', age: 28 },
  { id: 2, name: 'Bob', age: 35 },
  { id: 3, name: 'Charlie', age: 42 },
  { id: 4, name: 'Diana', age: 31 },
];

class UserRow extends Component {
  render() {
    console.log(`Rendering PURE Row for: ${this.props.user.name}`);
    return (
      <tr>
        <td>{this.props.user.id}</td>
        <td>{this.props.user.name}</td>
        <td>{this.props.user.age}</td>
      </tr>
    );
  }
}

function App() {
  const [users, setUsers] = useState(initialUsers);

  return (<div className="App">
            <tbody>
              {users.map(user => (
                <UserRow key={user.id} user={user} />
              ))}
          </tbody>
        </div>);
}
```

### Some tricks using DOM tree diffs to improve rendering performance

The primary mechanism React uses to track elements and optimize its diffing process is the `key` attribute.
For long array and nested objects, smart key design is the key to cater to the "diffing" process.

For example,
without `key`, React will, by default, use the array index as the key.
When the list is modified. React will assume that every subsequent element has changed because their indices have shifted.
The updates are heavy.

Bad implementation:

```js
// Unstable and can lead to issues
const items = ['Apple', 'Banana', 'Cherry'];
items.map((item, index) => <li key={index}>{item}</li>);
```

Good Practice: Using a Stable ID from Data

```js
// Stable and unique IDs from data
const items = [
  { text: 'Apple' },
  { text: 'Banana' },
  { text: 'Cherry' },
];
items.map(item => <li key={item.text}>{item.text}</li>);
```

For complex nested objects, need to manually construct semantic ids as keys.

```js
const categories = [
  {
    id: 'fruit1',
    name: 'Fruits',
    items: [
      { id: 'item1', name: 'Apple' },
      { id: 'item2', name: 'Banana' },
    ],
  },
];

const CategoryList = () => (
  <ul>
    {categories.map(category => (
      <li key={category.id}>
        <h3>{category.name}</h3>
        <ul>
          {category.items.map(item => (
            // A composite key ensures uniqueness across all inner list items
            <li key={`${category.id}-${item.id}`}>{item.name}</li>
          ))}
        </ul>
      </li>
    ))}
  </ul>
);
```

## How to check and improve React performance

### `pureComponent`

By default, a standard React.Component re-renders whenever its parent component re-renders, or when its own state or props change, regardless of whether the new state or props are identical to the old ones. 

`React.PureComponent` addresses this by implementing the `shouldComponentUpdate()` lifecycle method with a "shallow comparison" of its current and next props and state.

#### Shallow Comparison Mechanism

* For primitive types (like strings, numbers, and booleans), a shallow comparison checks for value equality.
* For complex types (like objects and arrays), a shallow comparison checks for reference equality. **NOT for nested objects**.

#### Example Use Case

In a requirement of grid table, where user may just edit one entry but depending on component dependency chain, the whole parent grid component is updated that has a cascading effect on all child rows.
This is expensive.

In the example provided `UserRow` vs `PureUserRow` in `./UserRow`,

```js
import React, { Component, PureComponent } from 'react';

class UserRow extends Component {
  render() {
    console.log(`Rendering PURE Row for: ${this.props.user.name}`);
    return (
      <tr>
        <td>{this.props.user.id}</td>
        <td>{this.props.user.name}</td>
        <td>{this.props.user.age}</td>
      </tr>
    );
  }
}

class PureUserRow extends PureComponent {
  render() {
    console.log(`Rendering PURE Row for: ${this.props.user.name}`);
    return (
      <tr>
        <td>{this.props.user.id}</td>
        <td>{this.props.user.name}</td>
        <td>{this.props.user.age}</td>
      </tr>
    );
  }
}
```

Consider a grid table of four rows.
When user clicks a button to update Bob's age, every `<UserRow key={user.id} user={user} />` is re-rendered, while for `<PureUserRow key={user.id} user={user} />` only the Bob's entry is re-rendered.

This (the standard `UserRow` scenario) is triggered by the state `users` of `App` updated as a parent, whose children the four `UserRow` are updated by a cascading effect:

1. State Change: User clicks the button, which calls `setUsers`. This updates the state of the `App` component with a new array of users.
2. Parent Re-renders: Because `App`'s state has changed, the `App` component's `render()` method is executed again.
3. If a parent component (`App`) re-renders, React will re-render all of its child components by default.

The above issue can be addressed by child component implementing `PureComponent`.

```js
import React, { useState } from 'react';
import { UserRow, PureUserRow } from './UserRow';
import './App.css';

const initialUsers = [
  { id: 1, name: 'Alice', age: 28 },
  { id: 2, name: 'Bob', age: 35 },
  { id: 3, name: 'Charlie', age: 42 },
  { id: 4, name: 'Diana', age: 31 },
];

function App() {
  const [users, setUsers] = useState(initialUsers);
  
  const updateUser = () => {
    setUsers(prevUsers => {
      // Create a new array to ensure state immutability
      const newUsers = [...prevUsers];
      // "Update" Bob's age
      newUsers[1] = { ...newUsers[1], age: newUsers[1].age + 1 };
      return newUsers;
    });
  };

  return (
    <div className="App">
      <button onClick={updateUser}>Update Bob's Age</button>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Age</th>
          </tr>
        </thead>
        <tbody>
          {users.map(user => (
            <UserRow key={user.id} user={user} />
          ))}
        </tbody>
        <tbody>
          {users.map(user => (
            <PureUserRow key={user.id} user={user} />
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

### Chrome V8 Engine Garbage Collection (GC)

### Worker, thread worker

JavaScript is single-threaded, that means at its core, JavaScript has a single call stack, which is a data structure that records where in the program the execution is.

However, for heavy tasks, JS spawns worker threads.
The worker threads have below limitations out of main thread safety concerns.

* No DOM access: For safety and to prevent race conditions, Web Workers cannot directly manipulate the Document Object Model (DOM).
* Communication via messaging: The main thread and a worker thread communicate by sending messages to each other using the `postMessage()` method and responding to them via the onmessage event handler. Data is copied, not shared, between the threads.

Write a heavy workload task `/heavyTask.worker.js`

```js
// eslint-disable-next-line no-restricted-globals
self.onmessage = function(event) {
  console.log("Worker: Message received from main script");
  const number = event.data;
  // Simulate a CPU-intensive task
  let result = 0;
  for (let i = 0; i < number * 200000000; i++) {
    result += Math.sqrt(i);
  }
  console.log("Worker: Posting message back to main script");
  postMessage(result);
};
```

In the main thread, the worker thread is launched via `new Worker(...)` when `HeavyTaskComponent` is mounted awaiting user action.
When user inputs number and clicks button that triggers `handleCalculate` that send the input num via `workerRef.current.postMessage(inputNumber);`, the worker thread on receiving the event msg starts execution.

When the spawned worker thread finishes its computation, the main thread receives completion signal via `workerRef.current.onmessage`.

```js
function HeavyTaskComponent() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [inputNumber, setInputNumber] = useState(5);

  // useRef to hold the worker instance so it persists through re-renders
  const workerRef = useRef(null);

  useEffect(() => {
    // Create a new worker
    workerRef.current = new Worker('/heavyTask.worker.js');

    // Set up the listener for messages from the worker
    workerRef.current.onmessage = (event) => {
      console.log("Main: Message received from worker");
      setResult(event.data);
      setIsLoading(false);
    };

    // Clean up the worker when the component unmounts
    return () => {
      console.log("Main: Terminating worker");
      workerRef.current.terminate();
    };
  }, []); // Empty dependency array ensures this runs only once on mount

  const handleCalculate = () => {
    setIsLoading(true);
    setResult(null);
    console.log("Main: Posting message to worker");
    workerRef.current.postMessage(inputNumber);
  };

  return(
    <div>
      <input
        type="number"
        value={inputNumber}
        onChange={(e) => setInputNumber(parseInt(e.target.value, 10))}
        disabled={isLoading}
      />
      <button onClick={handleCalculate} disabled={isLoading}>
        {isLoading ? 'Calculating...' : 'Start Heavy Calculation'}
      </button>
    </div>);
  }
```

## What is HOC `highOrderComponent`

High order component is basically a wrap component to make easy for code reuse.

For example, there is `withBackgroundColor` that defines some generic styles.
Instead writing css for every individual component, make `withBackgroundColor` the wrapper component.

```js
import React from 'react';

const withBackgroundColor = (WrappedComponent) => {
  // The new component returned by the HOC
  const WithBackgroundColor = (props) => {
    const style = {
      backgroundColor: 'lightblue',
      padding: '10px',
      borderRadius: '5px'
    };

    // Render the original component with the new style prop
    return (
      <div style={style}>
        <WrappedComponent {...props} />
      </div>
    );
  };

  return WithBackgroundColor;
};

export default withBackgroundColor;
```

For an individual component,

```js
import React from 'react';

const MyComponent = ({ message }) => {
  return (
    <div>
      <h1>Hello from MyComponent!</h1>
      <p>{message}</p>
    </div>
  );
};

export default MyComponent;
```

wrap it within `withBackgroundColor`.

```js
import React from 'react';
import MyComponent from './MyComponent';
import withBackgroundColor from './withBackgroundColor';

// Create the enhanced component by calling the HOC with the original component
const EnhancedComponent = withBackgroundColor(MyComponent);

// Use the new component in your application
const App = () => {
  return (
    <div>
      <EnhancedComponent message="This component has a background color from the HOC." />
    </div>
  );
};

export default App;
```
