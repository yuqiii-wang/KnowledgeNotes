# React

React is used to prevent full webpage change by DOM partial update.

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
