# Web Crawling

A Web crawler, sometimes called a spider or spiderbot and often shortened to crawler, is an Internet bot that systematically browses the World Wide Web.

* A **ChromeDriver** is a standalone server that implements the open-source Selenium WebDriver Chromium Protocol. The Selenium tests interact with the ChromeDriver using the JsonWireProtocol, which translates the Selenium commands into corresponding actions on the Chrome browser.

* Puppeteer vs Selenium

* Diffs between running js click() and WebDriver click()

webdriver best simulate a user's click, such as if an element `<A>` is covered by a parent `dev`, the parent element got clicked as well, while js click is a directly triggering of click event on `<A>` 

```py
# 
element = driver.find_element_by_id("myid")
driver.execute_script("arguments[0].click();", element)

#
element.click()
```
* presenceOfElementLocated vs visibilityOfElementLocated

`presenceOfElementLocated` will be slightly faster because it's just check that an element is present on the DOM of a page. This does not necessarily mean that the element is visible. while the `visibilityOfElementLocated` has to check that an element is present on the DOM of a page and visible (got width and height).

## Selenium Configurations

* Page loading strategy:

```py
from selenium.webdriver.chrome.options import Options
options = Options()
options.page_load_strategy = 'normal' 
# options.page_load_strategy = 'eager' 
# options.page_load_strategy = 'none' 
```

set to `normal`, Selenium WebDriver waits until the load event fire is returned.

set to `eager`, Selenium WebDriver waits until DOMContentLoaded event fire is returned (discarded loading of stylesheets, images and subframes).

set to `none` Selenium WebDriver only waits until the initial page is downloaded.