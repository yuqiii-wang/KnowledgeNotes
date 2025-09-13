# APP Development

## APP UI Interaction Concepts

### Fingertip vs. The Click

* Fingertip: `touchstart` and `touchend`
* Click: `mouseup` and `mousedown`

The most significant divergence lies in their capacity for **multi-touch** interactions.
Touch events are inherently designed to handle multiple simultaneous points of contact.

Touch events are always tied to the element where the touch began.
In contrast, mouse events target the element currently under the cursor.

If UI not supported touch, e.g., old browser, the underlying behaves in response to q single tap on a screen:

1. `touchstart`, `touchend`
2. (after a slight delay) `mousedown`, `mouseup`
3. finally a `click` event.
