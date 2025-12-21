# Advanced Vue Knowledge

## Manual Rendering Control

In Vue, state changes are synchronous, but DOM updates are asynchronous.
For example, when there is a change in state (e.g., `this.someData = 'new value'`), Vue does not update the DOM immediately. Instead, it starts a process:

1. Detection: Vue's reactivity system (Proxies in Vue 3) detects the data change.
2. Queueing: Vue adds the component explicitly to a **scheduler queue**.
3. Buffering & Deduplication: Vue buffers the changes and only performs one update for the final state. If the same change to data is made 10 times in a row, Vue won't re-render the component 10 times.
4. Flushing (The "Tick"): Once the synchronous code (your function) finishes running, Vue "flushes" the queue. It calculates the Virtual DOM diff and patches the actual browser DOM.

### What is `nextTick`

`nextTick` is a utility that allows **execute code immediately after the DOM update cycle completes**.

This is helpful in the scenario when a component is not yet ready but was already written in script code to update.
Since the component loading/mounting sequence might be in disorder, this updated state might not trigger re-render or even could throw error.

`nextTick` can help on component ready on the next tick to render the state change.

```ts
<script setup>
import { ref, nextTick } from 'vue';

const isVisible = ref(false);
const inputField = ref(null);

const showAndFocus = async () => {
  // 1. on state change,
  // Vue queues a DOM update, but the <input> is NOT created yet.
  isVisible.value = true;

  // ❌ BAD: This would crash or do nothing.
  // inputField.value is still null here!
  // inputField.value.focus(); 

  // 2. We wait for the DOM update cycle to finish.
  await nextTick();

  // 3. Now the DOM is updated, <input> exists, and the ref is filled.
  // ✅ GOOD: This works safely.
  inputField.value.focus();
};
</script>

<template>
  <button @click="showAndFocus">Show Search</button>
  
  <!-- This element does not exist in the DOM initially -->
  <input 
    v-if="isVisible" 
    ref="inputField" 
    placeholder="Type here..." 
  />
</template>
```
