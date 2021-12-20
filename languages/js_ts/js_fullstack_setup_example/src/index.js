const express = require('express')
const path = require('path')
const getMongoData = require('./getMongoData')

const app = express()
const port = 8080

app.set('view engine', 'ejs');
app.use(express.static('static'));

app.get('/', (req, res) => {
  
  var runIt = async (req, res) => {
    var queryRes = await getMongoData(); 
    var timestamp_arr = queryRes[0];
    var val_arr = queryRes[1];
    res.render('index', {timestamp_arr:timestamp_arr, 
                        val_arr:val_arr});
  }
  runIt(req, res);
})

app.get('/echarts.min.js', (req, res) => {
  res.render('echarts.min.js')
})

app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`)
})