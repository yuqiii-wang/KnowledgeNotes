var MongoClient = require('mongodb').MongoClient;

const getMongoData = async function () {
  var timestamp_arr = [];
  var prodcode_arr = [];
  var lastPrice_arr = [];
  var lastQty_arr  =[];

  var url = "mongodb://<localhost_mongo_server>";

  var now = new Date();
  var now_tick = now.getTime() / 1000;
  var lastWeek_tick = now_tick - 3600 * 24 * 14;

  var mgClient = new MongoClient(url);
  await mgClient.connect();
  var dbo = mgClient.db("<localhost_mongo_db_name>");
  var query = { "timestamp": {"$gt": lastWeek_tick, "$lt": now_tick}} // get past two week data
  var docs = await dbo.collection("<localhost_mongo_collection_name>").find(query);
  
  await docs.forEach(element => {
    timestamp_arr.push(element.timestamp);
    val_arr.push(element.value);
  });
  mgClient.close();

  return [timestamp_arr, val_arr];
}

module.exports = getMongoData;