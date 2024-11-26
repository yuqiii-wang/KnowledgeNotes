# React Practices

## React Quick Start

### Optional: for China user

Choose one of the below as mirror (for npm)

* `npm config set registry https://registry.npm.taobao.org`
* `npm install -g cnpm --registry=https://registry.npm.taobao.org`

where if by `cnpm`, all next `npm` be replaced with `cnpm`.
For example, instead by `npm init react-app my-app`, go with `cnpm init react-app my-app`.

### Create a new React app

Run one of the below to launch a new react app (remember to replace `npm` with `cnpm` if in China).

* `npx create-react-app my-app`
* `npm init react-app my-app`

### React @17 vs @18 Version

* React 17 example:

```js
import React from "react";
import { render } from "react-dom";
import "./index.css";
import App from "./App";

const root = document.getElementById("root");
render(<App />, root);
```

* React 18 example:

```js
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

Reference:
https://stackoverflow.com/questions/71913692/module-not-found-error-cant-resolve-react-dom-client

## React syntax and Cautions

* Do NOT use hooks, e.g., `useState` as an intermediary shared data between functions, but only pass variable.

## React and Bootstrap

Reference:
https://react-bootstrap.netlify.app/docs/getting-started/introduction

*Bootstrap* is one of the most popular frontend layout packages.

`react-bootstrap` might not always be sync with `react`, so that using lower `react` version is advised.
By 08 Aug 2024, the latest `react` is `@18`, but `react-bootstrap@2.10.4` is compatible with `react@17.0.2`.

### Quick Start

Run `npm install react-bootstrap bootstrap`.

Add `import 'bootstrap/dist/css/bootstrap.min.css';` in `src/index.js` or `App.js` file.

## Hook

Hooks in react are "state" management, that upon state change, react re-renders the change accordingly.
React only re-renders the state-related components.

* `useState`: only in function-scope

In this example, the component `<p>Count: {count}</p>` changes on UI whenever user presses a button `onClick={() => setCount(count + 1)}`.

```js
import React, { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
    </div>
  );
}
```

* `useEffect`: for side effects

It can be said it got triggered on another state change.

For example, when a user presses a button `onClick={() => setCount(count + 1)}`, the state of `count` changes; then, `useEffect` got triggered for it takes `[count]` as the dependent variable: `useEffect(() => {...}, [count])`.

```js
import React, { useState, useEffect } from 'react';

function ButtonPressLogger() {
  const [count, setCount] = useState(0);

  // useEffect to run a side effect whenever 'count' changes (when button is pressed)
  useEffect(() => {
    console.log(`Button pressed ${count} times`);

    // Optionally, you can add cleanup logic here
    return () => {
      console.log(`Cleanup on count ${count}`);
    };
  }, [count]); // This effect depends on the 'count', runs when 'count' changes

  return (
    <div>
      <p>Button has been pressed {count} times</p>
      <button onClick={() => setCount(count + 1)}>Press me</button>
    </div>
  );
}

export default ButtonPressLogger;
```

P.S., if there is no dependent variable such that `useEffect(() => {...}, [])`, the empty array means this will run only on component mount/unmount.

* `useContext`

`useContext` allows access values from a context in React.
`context` means a larger state scope than `useState`'s function state.
It can be used as global states or any custom cross-function scope.

To use it,

1. by `createContext` create a context, inside which some states are defined.
2. define `<*.Provider>` scope in a parent component where the context will be applied, that only in this parent component scope the context states are used.
3. from a child function/component set up `useContext` that loads the state.
  
```js
import React, { createContext, useContext } from 'react';

const ThemeContext = createContext('light');

function DisplayTheme() {
  const theme = useContext(ThemeContext);
  return <p>Current theme: {theme}</p>;
}

function App() {
  return (
    <ThemeContext.Provider value="dark">
      <DisplayTheme />
    </ThemeContext.Provider>
  );
}
```

## StrictMode in React