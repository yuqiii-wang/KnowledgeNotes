# EJS

EJS (Embedded JavaScript) templating is used with nodejs backend to render html (and not only to html, also frontend xml scripting) template.

Simply speaking, EJS is a html template.

## Start with

Install ejs
```bash
npm install ejs
```

Render HTML code
```js
let ejs = require('ejs');
let people = ['geddy', 'neil', 'alex'];
let html = ejs.render('<%= people.join(", "); %>', {people: people});
```

The generated html can be used by backend such as express as view for frontend page rendering.

`views/index.ejs`:
```js
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
  </head>
  <body>
    <script type='text/javascript'>
      var valArr = [<%= val_arr %>];
    </script>
  </body>
</html>
```

`index.js` run on express:
```js
const app = express()
app.set('view engine', 'ejs');

var val_arr = [1,2,3];
app.get('/', (req, res) => {
    res.render('index', {val_arr:val_arr});
})
```