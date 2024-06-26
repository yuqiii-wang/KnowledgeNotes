# Redis Example Uses

## Practices

Remote connect (Do not use `-a` if there is sensitive data concerned):
```bash
redis-cli -h ${host} -p ${port} -a ${password}
```

Populate data:
```bash
# run four times
4 INCR visit

HMSET visit:1 url www.baidu.com timestamp 1 country CN user_id 1
HMSET visit:2 url www.baidu.com timestamp 1 country CN user_id 2
HMSET visit:3 url www.google.com timestamp 3 country CN user_id 3
HMSET visit:4 url www.google.com timestamp 4 country US user_id 4

ZADD visit-timestamp 1 visit:1
ZADD visit-timestamp 1 visit:2
ZADD visit-timestamp 3 visit:3
ZADD visit-timestamp 4 visit:4

ZADD visit.timestamp.index 1 1
ZADD visit.timestamp.index 3 2
ZADD visit.timestamp.index 4 3

# Verify the populated data
ZRANGE visit-timestamp 0 999
```

```bash
FT.AGGREGATE visit.timestamp.index "*"
  APPLY "@timestamp - (@timestamp % 3600)" AS hour
  
  GROUPBY 1 @hour
  	REDUCE COUNT_DISTINCT 1 @user_id AS num_users
  	
  SORTBY 2 @hour ASC
```

## Project-wise Solution

### Autocomplete

In the web world, autocomplete is a method that allows us to quickly look up things that
we want to find without searching. Generally, it works by taking the letters that we’re
typing and finding all words that start with those letters.

* Solution:

Use `LIST` to push/pop to add/remove recent searched words; Each item is a `String` so that substring compare can be used to match prefix.

### Cloud distributed locks

In Cloud app development, services are micro modules interacting with each other. There should be locks to prevent service-level concurrency.

* Solution:

Use `String` to set bool value to indicate if there is a lock. See Redlock for reference.

### Tick data streaming

Redis can be used to streaming data by `subscribe` and `publish`.

A client can subscribe a channel named `A-Share-Index`
```bash
SUBSCRIBE A-Share-Index
```

A message sender can publish messages on a channel
```bash
PUBLISH A-Share-Index 3000
PUBLISH A-Share-Index 2999
PUBLISH A-Share-Index 2998
```

### JSON data handling

Use RedisJSON; RedisJSON is a module that provides a special datatype and direct manipulation commands. RedisJSON stores the data in a binary format which removes the storage overhead from JSON, provides quicker access to elements without de-/re-serialization times.

Example:
```bash
JSON.SET car:1 $ '{"colour":"blue","make":"saab","model":93,"features":["powerlocks","moonroof"]}'
```

### Time series data handling

Handle time series by `ZSET`. However, `ZSET` only allows unique values. We can add entropy appending to least significant digits, such as Celsius $21.0$ turned into $21.0000017643$
```bash
# ZADD temperature(key) timestamp(ms) degree(Celsius)
ZADD temperature 1511533205001 21.0000017643
ZADD temperature 1511533206001 21.0000091264
ZADD temperature 1511533207001 21.0000045398
```

## C++ Load and Get

```cpp
void RedisClient::storeTrade(BasicData::Trade& trade)
{
    std::stringstream autoIncrTrade;
    autoIncrTrade << "INCR " << trade.productName << "-trade";
    
    redisReply* reply = (redisReply *)redisCommand(redisSession, autoIncrTrade.str().c_str());
    if (reply->type == REDIS_REPLY_ERROR)
    {
        redisLoggerPtr->error(reply->str);
        redisLoggerPtr->error(autoIncrTrade.str().c_str());
        sleep(1);
        freeReplyObject(reply);
        std::runtime_error("redis increment error");
    }

    std::stringstream getIncrTrade;
    getIncrTrade << "GET " << trade.productName << "-trade";
    
    std::stringstream tradeKey;
    reply = (redisReply *)redisCommand(redisSession, getIncrTrade.str().c_str());
    if (reply->type == REDIS_REPLY_ERROR)
    {
        redisLoggerPtr->error(reply->str);
        redisLoggerPtr->error(getIncrTrade.str().c_str());
        sleep(1);
        freeReplyObject(reply);
        std::runtime_error("redis get increment error");
    }
    tradeKey << trade.productName << "-trade" << ":" 
    << reply->str;


    std::stringstream setTradeHashSet;
    setTradeHashSet << "HMSET " <<  tradeKey.str() 
    << " productName " << trade.productName
    << " productPrimitiveName " << trade.productPrimitiveName 
    << " lastTradedPrice " << trade.lastTradedPrice 
    << " lastTradedQty " << trade.lastTradedQty 
    << " lastTradedTimestamp " << trade.lastTradedTimestamp 
    << " lastTradedTimestampMili " << trade.lastTradedTimestampMili ;
    reply = (redisReply *)redisCommand(redisSession, setTradeHashSet.str().c_str());
    if (reply->type == REDIS_REPLY_ERROR)
    {
        redisLoggerPtr->error(reply->str);
        redisLoggerPtr->error(setTradeHashSet.str().c_str());
        sleep(1);
        freeReplyObject(reply);
        std::runtime_error("redis setTradeHashSet error");
    }

    std::stringstream zaddTradeIndexing;
    zaddTradeIndexing << "ZADD " << trade.productName << "-trade-timestamp "
    << trade.lastTradedTimestamp << " " << tradeKey.str() ;
    reply = (redisReply *)redisCommand(redisSession, zaddTradeIndexing.str().c_str());
    if (reply->type == REDIS_REPLY_ERROR)
    {
        redisLoggerPtr->error(reply->str);
        redisLoggerPtr->error(zaddTradeIndexing.str().c_str());
        sleep(1);
        freeReplyObject(reply);
        std::runtime_error("redis zaddTradeIndexing error");
    }

    freeReplyObject(reply);
}

std::vector<BasicData::Trade> RedisClient::getTradesByTimestamp(
            std::string productName, long startTimestamp, long endTimestamp)
{
    std::vector<BasicData::Trade> tradeVec;

    std::stringstream rangeStr;
    rangeStr << "ZRANGEBYSCORE " << productName << "-trade-timestamp "
    << startTimestamp << " " << endTimestamp;
    redisReply* reply = (redisReply *)redisCommand(redisSession, rangeStr.str().c_str());
    if (reply->type == REDIS_REPLY_ERROR)
    {
        redisLoggerPtr->error(reply->str);
        redisLoggerPtr->error(rangeStr.str().c_str());
        sleep(1);
        freeReplyObject(reply);
        std::runtime_error("redis ZRANGE error");
    }

    int replyNum = reply->elements;
    tradeVec.reserve(replyNum);
    for (int idx = 0; idx < replyNum; idx++)
    {
        std::string tradeKey = reply->element[idx]->str;
        std::stringstream hgetallStr;
        hgetallStr << "HGETALL " << tradeKey;
        
        redisReply* replySub = (redisReply *)redisCommand(redisSession, hgetallStr.str().c_str());

        if (replySub->element == nullptr)
        {
            redisLoggerPtr->info("Empty str: {0}", hgetallStr.str());
            continue;
        }

        if (reply->type == REDIS_REPLY_ERROR)
        {
            redisLoggerPtr->error(replySub->str);
            redisLoggerPtr->error(hgetallStr.str().c_str());
            sleep(1);
            freeReplyObject(replySub);
            std::runtime_error("redis HGETALL error");
        }

        BasicData::Trade trade;
        int replySubNum = replySub->elements;
        for (int idx2 = 0; idx2 < replySubNum; idx2++)
        {
            if (idx2 % 2 == 0)
            {
                if ( !strcmp(replySub->element[idx2]->str, "productName"))
                {
                    strcpy(trade.productName, replySub->element[idx2+1]->str);
                }
                else if ( !strcmp(replySub->element[idx2]->str, "productPrimitiveName") )
                {
                    strcpy(trade.productPrimitiveName, replySub->element[idx2+1]->str);
                }
                else if ( !strcmp(replySub->element[idx2]->str, "lastTradedPrice") )
                {
                   trade.lastTradedPrice = std::stod( replySub->element[idx2+1]->str, 0 );
                }
                else if ( !strcmp(replySub->element[idx2]->str, "lastTradedQty") )
                {
                    trade.lastTradedQty = std::stoi( replySub->element[idx2+1]->str, 0 );
                }
                else if ( !strcmp(replySub->element[idx2]->str, "lastTradedTimestamp"))
                {
                    trade.lastTradedTimestamp = std::stol( replySub->element[idx2+1]->str, 0 );
                }
                else if ( !strcmp(replySub->element[idx2]->str, "lastTradedTimestampMili") )
                {
                    trade.lastTradedTimestampMili = std::stoi( replySub->element[idx2+1]->str, 0 );
                }
            }
        }

        freeReplyObject(replySub);

        tradeVec.emplace_back(trade);
    }

    freeReplyObject(reply);

    return tradeVec;
}

```