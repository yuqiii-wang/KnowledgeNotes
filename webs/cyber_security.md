# Cyber Security

**P.S. Good tutorials: WebGoat and WebWolf**

* SQL injection

If a user inputs `'Smith' OR TRUE;--'` to `SELECT * FROM users WHERE name=`, when executed, `SELECT * FROM users WHERE name='Smith' OR TRUE;--'` always returns all entries.

Solution: 
1. do input validation, disallow special characters
2. do not use direct SQL variable replacements, e.g., `'SELECT * FROM users WHERE name=' + users.getUserName() + ';'`

* XXE (XML External Entities)

An XML Entity allows tags to be defined that will be replaced by content when the XML Document is parsed.

An XML External Entity attack is a type of attack against an application that parses XML input. This attack occurs when XML input containing a reference to an external entity is processed by a weakly configured XML parser. 

In the example below, `js` is replaced with `passwd` when parsed, that contains user's passwords.
```xml
<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE author [
  <!ENTITY js SYSTEM "file:///etc/passwd">
]>
<author>&js;</author>
```

* Direct Object References

When using `GET` to retrieve `json` reponse, you can go to (for example in FireFox, `Inspect Elements -> Network -> (Click one request) -> Response`) Response to see all returned data, then guess possible server `path` and directly access unauthorised data.

For example, for an `submit` buttom, upon clicking it, you take action `GET http://localhost:8080/WebGoat/IDOR/profile/%7BuserId%7D`, (`%7BuserId%7D` translates to `{userId}`). Given a current login userId (usually the next userId to the current userId should be close to this one), we can kinda guess the next userId. In the code below, we run 100 times to guess what the next userId starting from the current userId.

```js
var httpRequest = new XMLHttpRequest();

var userCurrentID = 2342384;
for (var i = userCurrentID; i < userCurrentID + 100; i++)
{
  var urlLink = 'http://localhost:8080/WebGoat/IDOR/profile/' + i;
  console.log(urlLink)
  httpRequest.open("GET", urlLink, false); // should be `false` as XMLHttpRequest.open by default is async
  httpRequest.onload = function (e) {
    if (httpRequest.readyState === 4) {
      if (httpRequest.status === 200) {
        console.log(httpRequest.responseText);
      } else {
        console.error(httpRequest.statusText);
      }
    }
  };
  httpRequest.onerror = function (e) {
    console.error(httpRequest.statusText);
  };
  httpRequest.send(null); 
}
```

* Cross-Site Scripting 

XSS attacks enable attackers to inject client-side scripts into web pages viewed by other users. A cross-site scripting vulnerability may be used by attackers to bypass access controls such as the same-origin policy.

For example, you can on web console run
```js
aleart(document.cookies);
```
to see stored cookies for the current session (Many web applications rely on session cookies for authentication between individual HTTP requests)

Assume on the client side, the html page tempts user to click a buttom that triggers JavaScript code execution.
```html
<h3>
Click to Download this picture<script> aleart(document.cookies); </script>
</h3>
```

Very often, JavaScript code is sourced from external webservers attempting to maliciously access or modify user's browser's info. 

* Broken Authentication

Exploite POST payload to modify data to take advantage. For example, server takes keywords of a POST form request and forgets about verifying the existence of elements, such as a payload `secQuestion0=&secQuestion1=&jsEnabled=1&verifyMethod=SEC_QUESTIONS&userId=yourid`, edited to be `jsEnabled=1&verifyMethod=SEC_QUESTIONS&userId=yourid`, by which the POST request bypasses frontend JavaScript verification mechanism and is sent to server.