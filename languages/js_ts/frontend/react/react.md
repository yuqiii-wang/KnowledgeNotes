# React

Consider a data-driven webpage.
Frequent back-and-forth data transmissions between the frontend page and backend server are expensive.
DOM manipulation results in full webpage update/reload.

React is used to prevent full webpage change by DOM partial update.

React relies on a virtual DOM, which is a copy of the actual DOM. 
React's virtual DOM is immediately reloaded to reflect this new change whenever there is a change in the data state. 
After which, React compares the virtual DOM to the actual DOM to figure out what exactly has changed.

```html
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8" />
    <title>Hello World</title>
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>

    <!-- Don't use this in production: -->
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  </head>
  <body>
    <div id="root"></div>
    <script type="text/babel">
    
      function MyApp() {
        return <h1>Hello, world!</h1>;
      }

      const container = document.getElementById('root');
      const root = ReactDOM.createRoot(container);
      root.render(<MyApp />);

    </script>
    <!--
      Note: this page is a great way to try React but it's not suitable for production.
      It slowly compiles JSX with Babel in the browser and uses a large development build of React.

      Read this page for starting a new React project with JSX:
      https://react.dev/learn/start-a-new-react-project

      Read this page for adding React with JSX to an existing project:
      https://react.dev/learn/add-react-to-an-existing-project
    -->
  </body>
</html>
```

### Babel

Babel is a toolchain that is mainly used to convert ECMAScript 2015+ code into a backwards compatible version of JavaScript in current and older browsers or environments.
In other words, transcript higher version JavaScript code into lower version JavaScript.

For example, arrow function is converted to JavaScript ES5 equivalent.

```js
// Babel Input: ES2015 arrow function
[1, 2, 3].map(n => n + 1);
```

```js
// Babel Output: ES5 equivalent
[1, 2, 3].map(function(n) {
  return n + 1;
});
```

* JSX and React

JSX is an addition to the JavaScript syntax which is a mixture of both HTML and JavaScript.
JSX is extensively used in React.

For example, the code below exhibits embedding HTML DOMs collectively as a return component.

```jsx
export default function DiceRoll(){
  const getRandomNumber = () => {
    return Math.ceil(Math.random() * 6);
  };

  const [num, setNum] = useState(getRandomNumber());

  const handleClick = () => {
    const newNum = getRandomNumber();
    setNum(newNum);
  };

  return (
    <div>
      Your dice roll: {num}.
      <button onClick={handleClick}>Click to get a new number</button>
    </div>
  );
};
```
