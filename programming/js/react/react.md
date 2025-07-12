# React

## DOM (Document Object Model)

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

## Babel

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

* React View

## File Extensions

|Feature|`.js`|`.ts`|`.tsx`|
|:---|:---|:---|:---|
|Language|JavaScript|TypeScript|TypeScript|
|JSX Support|Yes (with a tool like Babel)|No|Yes|
|Type Safety|No|Yes|Yes|
|Common Use|Basic React components|Non-component TypeScript logic|React components in TypeScript|


## React Native

## `redux`

## React Router

## Responsive Programming

Responsive programming in react means applications adapt dynamically to different screen sizes, device types, and user interactions.

For example, React-Bootstrap or Material-UI come with built-in `grid` systems that are responsive by default.
In the below example, each column takes up the full width on small screens (`xs={12}`) but only half the width on medium screens (`md={6}`).

```js
import { Container, Row, Col } from 'react-bootstrap';

function ResponsiveGrid() {
  return (
    <Container>
      <Row>
        <Col xs={12} md={6}>
          Column 1
        </Col>
        <Col xs={12} md={6}>
          Column 2
        </Col>
      </Row>
    </Container>
  );
}
```
