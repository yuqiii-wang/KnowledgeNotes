# Cascading Style Sheets (CSS)

## Syntax

### Scope

* Tag: `tagname { property: value; }`

```css
p {
  color: blue;
}
```

* `.` class: `.classname { property: value; }`

```css
.highlight {
  background-color: yellow;
}
```

* `#` id: `#idname { property: value; }`

```css
#header {
  font-size: 24px;
}
```

* `*` Universal/apply to all: `* { property: value; }`

```css
* {
  margin: 0;
  padding: 0;
}
```

* `:` and `::`: Pseudo-classes and Pseudo-elements:

Take effect on certain conditions or part of an element `selector:pseudo-class { property: value; }` or `selector::pseudo-element { property: value; }`.

```css
p:hover {
  color: red;
}
/* Changes the link color to red when hovered over. */

p::first-line {
  font-weight: bold;
}
/* Makes the first line of a paragraph bold. */
```

The above css applies to this `<p>` tag element.

```html
<p>This is the first line of the paragraph, and it will be bold.
The rest of the text in this paragraph will not be bold. This is just some additional text to show that only the first line is affected by the CSS rule.
When on hover, the whole paragraph will be red.</p>
```

### Naming Convention in React: Inline vs in `.css`

In React, to JavaScript objects to define the styles, the property names must follow JavaScript's naming conventions, which do not allow hyphens in object keys.

For example, in `.css`, there is

```css
margin-top: 10px;
```

But in a React component by inline style, there is

```jsx
const myStyle = {
  marginTop: '10px'
};

function MyComponent() {
  return <div style={myStyle}>This div has a 10px margin on top.</div>;
}
```

### Style Precedence

1. `!important`
2. Javascript editing style
3. inline style
4. external css, e.g., loaded from local `.css` file, external `<link>` or `<style>`
5. agent default: user browser default value

Example:

By default, user browser renders a hyperlink with a blue underline.

```html
<a href="#">Link</a>
```

It can get overwritten (as red) by loaded external css.

```css
/* styles.css */
a {
  color: red;
}
```

```html
<link rel="stylesheet" href="styles.css">
<a href="#">Link</a>
```

Inline: by directly instructing the style be red:

```jsx
<a href="#" style={{ color: 'yellow' }}>Link</a>
```

JavaScript has higher priority changing the style.

```html
<!DOCTYPE html>
<html>
<head>
  <script>
    document.addEventListener("DOMContentLoaded", function() {
      var style = document.createElement('style');
      style.innerHTML = 'a { color: purple; }';
      document.head.appendChild(style);
    });
  </script>
</head>
<body>
  <a href="#">Link</a>
</body>
</html>
```

`!important` always has the highest priority.

```css
/* styles.css */
a {
  color: blue !important;
}
```

Only inlined `!important` can overwrites external css `!important` styles.

```jsx
<a href="#" style={{ color: 'yellow !important' }}>Link</a>
```

## Size: `rem`, `px`, `vh`, and `%`

### `px`

* A `px` (pixel) is a fixed unit that represents a single dot on the screen. It is a precise measurement.
* Ideal for small elements like borders, icons, or detailed layouts where precision is necessary.
* It doesn't scale well with different screen sizes or user settings.

### `rem` (Root Em)

* `rem` is a relative unit that is based on the root element's `<html>` font size.
By default, 1 `rem` equals 16 pixels, but this can change if you set a different font size on the `<html>` element.
* If you change the root font size, it affects all elements using rem, allowing easy scaling of your entire design.

### `vh` and `vw`

* `vh` and `vw` are relative to the viewport's height and width. 1 `vh` equals `1%` of the viewport's height, and 1 vw equals `1%` of the viewport's width.
* Useful for designing elements that adapt to the viewport size, like full-screen backgrounds or sections.

### `%` (Percentage)

* `%` is a relative unit that is based on the parent element's size.

## Layout

### `position`

* `static` default value
* `relative` offset by its original position
* `absolute` offset by its parent position
* `fixed` offset by agent window, e.g., user browser window
* `sticky` first appears as `relative`-like, once reached certain conditions, shows as `absolute`-like

#### `z-index`

`z-index` is used when there are multiple items overlapping each other, that which items are to shown in the front.

`z-index` is used with specified non-`static` `position`.

Higher the val, higher priority shown in the front.

For example,

```css
.box1 {
  background-color: red;
  top: 50px;
  left: 50px;
  position: absolute;
  z-index: 1; /* lower z-index, got overlapped shown in the back */
}

.box2 {
  background-color: blue;
  top: 75px;
  left: 75px;
  position: absolute;
  z-index: 2; /* higher z-index, overlapping `box1` shown in the front */
}
```

#### `top` vs `margin-top` (as well as `left`, `bottom` and `right`)

`margin-top` property sets space between the element and the elements above it.

The `top` property specifies the distance from the top edge of the item container to the top edge of the positioned element.
`top` works with non-`static` `position` items.

### `flex`

* `display: flex` - define this container as flex
* `flex-direction` - define main axis, either row or col（`row`, `row-reverse`, `column`, `column-reverse`）。
* `justify-content` - align items all to the start, evenly split, put in the middle, or to the end（`flex-start`, `flex-end`, `center`, `space-between`, `space-around`, `space-evenly`）。
* `align-items` - （`stretch`, `flex-start`, `flex-end`, `center`, `baseline`）
* `flex-wrap` - if to start with a new line（`nowrap`, `wrap`, `wrap-reverse`）

For example,

```html
<div class="flex-container">
  <div class="item">1</div>
  <div class="item">2</div>
  <div class="item">3</div>
</div>
```

In css, having declared `display: flex`, use `justify-content` and `align-items` to place items horizontally and vertically.

```css
.flex-container {
  display: flex;
  justify-content: space-between; /* items evenly distributed (in row) */
  align-items: center; /* vertically centered */
  height: 100px;
  border: 1px solid black;
}

.item {
  padding: 20px;
  background-color: lightgreen;
}
```

### `grid`

## `clip-path` and Border

`clip-path` clips off an element and renders this element of a particular shape.

There are

* `circle()`
* `ellipse()`
* `inset()` (rectangle)
* `polygon()`

For example, `clip-path: polygon( <point-list> );` defines a polygon, where input pair `x% y%` defines an element's vertex.
`clip-path: polygon(50% 0%, 100% 100%, 0% 100%);` gives an up-pointed triangle:

* `50% 0%`: mid top vertex
* `0% 100%`: left bottom vertex
* `100% 100%`: right bottom vertex
