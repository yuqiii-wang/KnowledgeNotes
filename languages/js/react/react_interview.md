# Typical React Interview Questions

## React State vs Property

* State

State is local and private to the component where it is defined.

State is used for re-rendering on change.

* Props:

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

## React Component Lifecycle

## Inheritance in React/JavaScript (Prototype Chain)

### `bind` and `this`

### constructor: `super` and `props`

## Iteration

### How React uses mapping as a for loop

## Virtual DOM vs Actual DOM

### Some tricks using DOM tree diffs to improve rendering performance

### What is Fiber

## React Rendering

### How React responds on `onClick`

### React rendering `useState` on change

### React Rendering concerning Promise, and `async`/`await`

## How to check and improve React performance

### `shouldComponentUpdate`

### `pureComponent`

### Chrome V8 Engine Garbage Collection (GC)

### Worker, thread worker

## What is HOC `highOrderComponent`

## React Storage

### General practices concerning state management

* `hooks`+`context`
* `redux`

### Resilient File System (ReFS)

### Session vs Local Storage

### `usememo`