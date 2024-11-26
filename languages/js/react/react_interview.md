# Typical React Interview Questions

## React State vs Property

* State

State is local and private to the component where it is defined.

State is used for re-rendering on change.

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
}, []); // Empty dependency array ensures it runs once on mount, multiple times after every render, and once on unmount for cleanup
```

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