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
Every displayable page must be registered in pages.json
