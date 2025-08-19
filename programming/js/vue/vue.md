# Vue

## Intro

### Vite

Vite (pronounced "veet," the French word for "quick") is a high-performance equivalent to webpack.

## Basic Syntax

### Vue DSL

For vue version 3,

* `<template>`: HTML markup for component.
* `<script>`: JS Logic.
* `<style>`: CSS.

Example:

```ts
<template>
  <div class="greeting">
    {{ message }}
  </div>
</template>

<script>
export default {
  data() {
    return {
      message: 'Hello, Vue!'
    }
  }
}
</script>

<style scoped>
.greeting {
  color: blue;
}
</style>
```

Event handling:

```js
<button @click="handleClick">Click Me</button>
```

### Nested Components

Filename by PascalCase (upper camel case) indicates the component name (by default).
The component name can be overwritten by `export default`.

For example, in `src/components/SubmitButton.vue` defines

```ts
<template>
  <button class="submit-btn">Submit</button>
</template>

<script>
export default {
  // 1. The 'name' property inside the component
  name: 'SubmitButton'
}
</script>
```

The component is used in another dir `src/App.vue`

```ts
<template>
  <SubmitButton />
</template>

<script setup>
// 2. The import name matches the file name
import SubmitButton from './components/SubmitButton.vue';
</script>
```

To import css, there is

```ts
<style>
	@import "./common/uni.css";
	
	.uni-hello-text{
		color:#7A7E83;
	}
</style>
```

### Inside `export default`

This statement (`export default`) exports a JavaScript object that defines the component's options and logic.
For example,

```ts
<script>
export default {
  data() {
    return {
      counter: 0
    }
  },
  methods: {
    incrementCounter() {
      this.counter++;
    },
    greet(name) {
      alert('Hello, ' + name);
    }
  }
}
</script>
```

where

* `data` function returns an object containing the component's reactive state; in React, the equivalent is `useState`
* `methods` is an object containing functions that are typically used for event handlers or to encapsulate reusable logic within the component; in React, the equivalent is internal function

The above vue code is equivalent to the below React code.

```jsx
import React, { useState } from 'react';

function MyComponent() {
  const [counter, setCounter] = useState(0);

  const incrementCounter = () => {
    setCounter(counter + 1);
  };

  return <button onClick={incrementCounter}>Increment</button>;
}
```

#### Syntactic Sugar `<script setup>`

Since Vue 3, `<script setup>` is a compile-time syntactic sugar to make code concise to replace the below in script.

```ts
export default {
  setup() {
    ...
  }
}
```

### The `v-` Directives

The `v-` directives can manipulate DOMs by typical control logics such as `if`/`else` and `for`.

#### `v-if` and `v-else`

```ts
<template>
  <div>
    <button @click="toggleVisibility">Toggle Content</button>

    <div v-if="isVisible">
      <h2>This is visible!</h2>
      <p>You can see this content because 'isVisible' is true.</p>
    </div>
    <div v-else>
      <h2>This is hidden.</h2>
      <p>Click the button to show the content.</p>
    </div>

    <hr>

    <div v-if="type === 'A'">
      Content for A
    </div>
    <div v-else-if="type === 'B'">
      Content for B
    </div>
    <div v-else>
      Content for something other than A or B
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      isVisible: true,
      type: 'A'
    };
  },
  methods: {
    toggleVisibility() {
      this.isVisible = !this.isVisible;
    }
  }
};
</script>
```

#### `v-for`

```ts
<template>
  <div>
    <h2>My To-Do List</h2>
    <ul>
      <li v-for="item in items" :key="item.id">
        {{ item.text }}
      </li>
    </ul>

    <h2>User Information</h2>
    <ul>
      <li v-for="(value, key, index) in user" :key="key">
        {{ index }}. {{ key }}: {{ value }}
      </li>
    </ul>
  </div>
</template>

<script>
export default {
  data() {
    return {
      items: [
        { id: 1, text: 'Learn Vue.js' },
        { id: 2, text: 'Build a project' },
        { id: 3, text: 'Deploy the project' }
      ],
      user: {
        name: 'John Doe',
        email: 'john.doe@example.com',
        role: 'Developer'
      }
    };
  }
};
</script>
```

#### `v-model`

`v-model` provides two-way data binding between an input element (like `<input>`, `<textarea>`, `<select>`) and a data property in component's state.

For example,

```ts
<input v-model="text">
```

is much more concise than

```ts
<input
  :value="text"
  @input="event => text = event.target.value"
>
```

## Vue Component LifeCycle

### Vue Component LifeCycle Phases

#### Creation Phase

* `beforeCreate`/`setup()`: The component has been initialized, but its reactive data, computed properties, etc., are not yet set up.
* `created`/`setup()`: The component has now processed its options, meaning reactive data, computed properties, methods, and watchers are all set up.

#### Mounting Phase

* `onBeforeMount`: This hook is called right before the component is about to be mounted to the DOM.
* `onMounted`: This is one of the most commonly used hooks. It's called after the component has been mounted and its DOM tree has been created and inserted into the parent element.

#### Updating Phase

* `onBeforeUpdate`: Runs right after data has changed and before the component re-renders and patches the DOM.
* `onUpdated`: Called after a data change has caused the component to re-render and the DOM has been patched.

#### Unmounting Phase

* `onBeforeUnmount`: Called right before a component instance is unmounted and destroyed.
* `onUnmounted`: Called after the component has been completely unmounted and destroyed.

### Vue Component LifeCycle Example

```ts
<template>
  <div>
    <p>{{ message }}</p>
    <button @click="message = 'Updated Message!'">Update</button>
  </div>
</template>

<script>
import { ref, onBeforeMount, onMounted, onBeforeUpdate, onUpdated, onBeforeUnmount, onUnmounted } from 'vue';

export default {
  setup() {
    console.log('Component is being set up (replaces created)');

    const message = ref('Hello, Vue!');

    onBeforeMount(() => {
      console.log('onBeforeMount: Component is about to be mounted.');
    });

    onMounted(() => {
      console.log('onMounted: Component has been mounted to the DOM.');
      // Ideal place to access this.$el or other DOM elements
    });

    onBeforeUpdate(() => {
      console.log('onBeforeUpdate: Component is about to update due to data change.');
    });

    onUpdated(() => {
      console.log('onUpdated: Component has been updated and the DOM is patched.');
    });

    onBeforeUnmount(() => {
      console.log('onBeforeUnmount: Component is about to be unmounted.');
      // Cleanup timers, event listeners, etc. here
    });

    onUnmounted(() => {
      console.log('onUnmounted: Component has been unmounted.');
    });

    return {
      message
    };
  }
};
</script>
```

## Vue Hooks

### Hooks in Vue vs. in React

||React|Vue|
|:---|:---|:---|
|State Management|`useState`: Returns an array with the value and a setter function. `const [count, setCount] = useState(0)`|`ref` & `reactive`: `ref` is used for any value type and wraps it in an object with a `.value` property. `const count = ref(0)`|
|Side Effects & Lifecycle|On Mount: `useEffect(() => { ... }, [])`|On Mount: `onMounted(() => { ... })`|
||On Unmount/Cleanup: The returned function from useEffect. `useEffect(() => { return () => { ... } }, [])`|On Unmount/Cleanup: `onUnmounted(() => { ... })`|
||On Data Change: `useEffect(() => { ... }, [someValue])`|On Data Change: `watch(someValue, () => { ... })`|
|Memoizing Values and Functions|`useMemo`: Caches the result of a function.|`computed`: Creates a derived reactive reference. It automatically tracks its dependencies and re-calculates only when they change.|
||`useCallback`: Caches a function definition itself.|N/A|

### Common Vue Hooks

#### Reactivity Hooks (State Management)

* `ref()`: The workhorse for creating reactive variables. It takes a value (string, number, boolean, object) and returns a reactive object with a .`value` property.
  * Key Rule: must use `.value` to access or change its value inside the `<script>` block. In the `<template>`, Vue automatically "unwraps" it and renders it.
* `reactive()`: An alternative to ref that only works for objects (or arrays). It makes the object itself deeply reactive. Do NOT use `.value` with properties of a reactive object.
* `computed()`: Creates a reactive value that is derived from other reactive data. It's cached and only re-calculates when its dependencies change.

```ts
<script setup>
import { ref, reactive, computed } from 'vue'

// Using ref() for a primitive value
const count = ref(0);

function increment() {
  count.value++; // Must use .value
}

// Using reactive() for an object
const user = reactive({
  firstName: 'John',
  lastName: 'Doe'
});

// Using computed() for derived data
const fullName = computed(() => {
  return `${user.firstName} ${user.lastName}`;
});
</script>

<template>
  <p>Count: {{ count }}</p> <!-- No .value needed in template -->
  <button @click="increment">Increment</button>

  <p>Full Name: {{ fullName }}</p>
  <input v-model="user.firstName" /> <!-- No .value needed here -->
</template>
```

#### Watch Hook for Side Effects

* `watch()`: "Watches" a specific ref or reactive source and runs a callback function whenever the source changes.
* `watchEffect()`: Runs a function immediately and then re-runs it whenever any of its reactive dependencies change.

```ts
<script setup>
import { ref, watch, watchEffect } from 'vue'

const question = ref('');
const answer = ref('I cannot give you an answer until you ask a question!');

// `watch` is great for fetching data based on a specific input change
watch(question, async (newQuestion, oldQuestion) => {
  if (newQuestion.includes('?')) {
    answer.value = 'Thinking...';
    // Fake API call
    setTimeout(() => {
      answer.value = `The answer to "${newQuestion}" is 42.`;
    }, 1000);
  }
});

// `watchEffect` is simpler for logging or reacting to multiple dependencies
watchEffect(() => {
  // This will run whenever `question` or `answer` changes
  console.log(`The user's question is: ${question.value}`);
});
</script>

<template>
  <p>Ask a question (end with a '?'):</p>
  <input v-model="question" />
  <p>{{ answer }}</p>
</template>
```