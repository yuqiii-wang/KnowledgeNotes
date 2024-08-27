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

## React and Bootstrap

Reference:
https://react-bootstrap.netlify.app/docs/getting-started/introduction

*Bootstrap* is one of the most popular frontend layout packages.

`react-bootstrap` might not always be sync with `react`, so that using lower `react` version is advised.
By 08 Aug 2024, the latest `react` is `@18`, but `react-bootstrap@2.10.4` is compatible with `react@17.0.2`.

### Quick Start

Run `npm install react-bootstrap bootstrap`.

Add `import 'bootstrap/dist/css/bootstrap.min.css';` in `src/index.js` or `App.js` file.

