const express = require('express')
const path = require('path')
const app = express()
const port = 3000

app.set('view engine', 'ejs');
app.use(express.static('static'));

app.get('/', (req, res) => {
  res.render('index')
})

app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`)
})