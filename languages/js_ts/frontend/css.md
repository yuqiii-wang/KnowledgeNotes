# Cascading Style Sheets (CSS)

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