# Uni App

## Uni-App Intro

*uni-app* is a cross-platform framework for developing all front-end applications using `Vue.js`, that same code can run on iOS, Android, HarmonyOS Next, Web (responsive) and various MiniApp (weixin/alipay/baidu/douyin/lark).

### uni-app x ecosystem: UTS and uVue

* uni-app x: The evolution of the original uni-app framework, designed for higher performance and deeper integration with native platforms.
* UTS (uni TypeScript): A TypeScript-like programming language
* uVue: The rendering engine for uni-app x. While UTS handles the application's logic, uVue is responsible for rendering the user interface based on a Vue.js-like syntax.

### Syntax and Features

#### Vue.js Syntax

Uni-app uses the same template syntax, data binding, directives (like `v-if`, `v-for`), and component lifecycle hooks as Vue.js.

#### Inter-Compatible Components

Uni-app utilizes a set of built-in components that are optimized for cross-platform compatibility.
These components often have names similar to their HTML counterparts but are designed to render natively on different platforms. For example:

* `<view>`: Similar to `<div>`, used as a container for other elements.
* `<text>`: Similar to `<span>`, used for displaying text.
* `<image>`: Used for displaying images.
* `<button>`: A standard button component.
* `<navigator>`: For navigating between pages.

#### Inter-Compatible APIs

Uni-app APIs: Uni-app provides a rich set of APIs for accessing device functionalities and platform-specific features. These APIs are prefixed with `uni`. and are designed to work consistently across all supported platforms.

* `uni.request()`: For making network requests.
* `uni.navigateTo()`: For navigating to a new page.
* `uni.showToast()`: For displaying a toast message.

#### Pages and Configuration

Pages and Configuration: A uni-app project has a `pages.json` file that configures the application's pages, global styles, and other settings.
Every displayable page must be registered in `pages.json`

## Uni-APP vs Vue (Web)

### Builtin Components

|Vue.js (Web)|uni-app (Cross-Platform)|Purpose|
|:---|:---|:---|
|`<div>`|`<view>`|A container view, similar to a block element.|
|`<span>`, `<p>`|`<text>`|A text container, for inline text display.|
|`<img>`|`<image>`|An image component with platform-optimized loading.|
|`<input>`|`<input>`|An input field, with types adapted for mobile use.|
|`<a>`|`<navigator>`|A component for page navigation.|

### Uni-APP APIs

No DOM Access: cannot use document.getElementById() or other direct DOM manipulation methods.

Phone-friendly methods: `uni.scanCode()`, `uni.getLocation()`, `uni.showToast()`, etc., for accessing device hardware and native UI.

### Routing: `pages.json` vs. `vue-router`

Vue (Web) implements `vue-router` library for page routing.

Uni-APP implements `pages.json` for routing.

### CSS Styling

Uni-app promotes the use of `rpx` (responsive pixels).
It's a dynamic unit that adapts to different screen widths, making it easier to write responsive layouts for various devices.
The base is a `750px` wide screen, where `750rpx` equals the screen width.

## Uni-APP LifeCycle

### Uni-APP Lifecycle Hook Intro

Lifecycle Stage|Component Lifecycle Hook|Page Lifecycle Hook
Creation & Loading|beforeCreate, created|onLoad, onShow, onReady
Rendering|beforeMount, mounted|
Updates|beforeUpdate, updated|
Destruction|beforeDestroy, destroyed|onUnload

### LifeCycle Uni-APP vs Vue

#### 1. Creation Phase

| Vue | Uni-app | Purpose | Alternative |
|-----|---------|---------|-------------|
| `beforeCreate` | *No direct equivalent* | Component initializing, no data access | `onLoad() { /* early init */ }` |
| `created` / `setup()` | *No direct equivalent* | Data reactive, computed properties ready | `onLoad() { this.initData(); }` |

#### 2. Page Loading Phase (Uni-app Exclusive)

| Vue | Uni-app | Purpose | Alternative |
|-----|---------|---------|-------------|
| *Manual route handling* | `onLoad(options)` | Page first loads with route params | `created() { this.bookId = this.$route.params.id; }` |

#### 3. Mounting Phase
| Vue | Uni-app | Purpose | Alternative |
|-----|---------|---------|-------------|
| `onBeforeMount` | *No direct equivalent* | Just before DOM mounting | `onReady() { /* pre-render setup */ }` |
| `onMounted` | `onReady` | DOM ready, can query elements | `mounted() { this.$nextTick(() => this.setupDOM()); }` |

#### 4. Visibility Phase (Uni-app Exclusive)

| Vue | Uni-app | Purpose | Alternative |
|-----|---------|---------|-------------|
| *Manual visibility detection* | `onShow()` | Page becomes visible | `mounted() { document.addEventListener('visibilitychange', this.handleShow); }` |

#### 5. Updating Phase (Runtime)

| Vue | Uni-app | Purpose | Alternative |
|-----|---------|---------|-------------|
| `onBeforeUpdate` | *No direct equivalent* | Before re-render | `watch: { data() { /* before update logic */ } }` |
| `onUpdated` | *No direct equivalent* | After re-render complete | `watch: { data() { this.$nextTick(() => /* after update */); } }` |

#### 6. Hidden Phase (Uni-app Exclusive)

| Vue | Uni-app | Purpose | Alternative |
|-----|---------|---------|-------------|
| *Manual visibility detection* | `onHide()` | Page becomes hidden | `beforeUnmount() { document.removeEventListener('visibilitychange', this.handleHide); }` |

#### 7. Unmounting Phase

| Vue | Uni-app | Purpose | Alternative |
|-----|---------|---------|-------------|
| `onBeforeUnmount` | `onUnload` | Cleanup before destruction | `beforeUnmount() { this.cleanup(); }` |
| `onUnmounted` | *No direct equivalent* | After destruction complete | `onUnload() { this.$nextTick(() => /* final cleanup */); }` |

#### Mobile-Specific Events (Uni-app Exclusive)

| Vue | Uni-app | Purpose | Alternative |
|-----|---------|---------|-------------|
| *Custom scroll handling* | `onPullDownRefresh()` | Pull to refresh | `mounted() { this.$refs.scroll.addEventListener('touchstart', this.handlePullDown); }` |
| *Custom scroll handling* | `onReachBottom()` | Scroll to bottom | `mounted() { this.$refs.scroll.addEventListener('scroll', this.checkBottom); }` |

## Uni-App Underlying Implementation

### `plus.io`

In uni-app, plus.io refers to the HTML5+ IO (Input/Output) module.
It is a set of native APIs provided by the DCloud (5+ Runtime) environment that bridge between JavaScript and the native OS (Android/iOS) file system.

It **only works on App platforms** (Android and iOS). It will not work on H5 (web browsers) or Mini Programs (WeChat, Alipay, etc.).

#### Directories (Sandbox)

To ensure security, the file system is sandboxed.

* `_www/`: The app's resource directory (where packaged code and static assets live). **Read-only**.
* `_doc/`: The app's private document directory. User can **read and write here**. Data persists after app restarts but may be cleared if the user clears app data.

