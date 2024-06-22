# Financial Information eXchange (FIX)

Financial Information eXchange (FIX) is a standard protocol for trade-related messages.

## FIX Structure

A FIX message consists of three main components:

* Header
* Body
* Tail

### Header

* `BeginString` (Tag 8): Identifies the beginning of a FIX message and the FIX version.
* `BodyLength` (Tag 9): Indicates the length of the message body.
* `MsgType` (Tag 35): Specifies the type of message (e.g., New Order, Execution Report).
* `SenderCompID` (Tag 49): The ID of the message sender.
* `TargetCompID` (Tag 56): The ID of the message receiver.
* `MsgSeqNum` (Tag 34): The sequence number of the message.
* `SendingTime` (Tag 52): The time the message was sent.

where

* Up to FIX.4.4, the header contains three fields: 8 (`BeginString`), 9 (`BodyLength`), and 35 (`MsgType`).
* From FIXT.1.1 / FIX.5.0, the header contains five or six fields: 8 (`BeginString`), 9 (`BodyLength`), 35 (`MsgType`), 49 (`SenderCompID`), 56 (`TargetCompID`) and the optional 1128 (`ApplVerID`).

### Body

Body is different per tag 35 `MsgType`.
For example, for New Order (MsgType = D):

* ClOrdID (Tag 11): Unique identifier for the order.
* Side (Tag 54): Side of the order (buy or sell).
* TransactTime (Tag 60): Time the order is submitted.
* OrderQty (Tag 38): Quantity of the order.
* OrdType (Tag 40): Type of order (market, limit, etc.).
* Price (Tag 44): Price for the order (if applicable).

### Tail

* `CheckSum` (Tag 10): Three-byte, simple checksum computed as the modulo 256 of the sum of all the bytes in the message.

## Example

### Example Msg

```csv
8=FIX.4.4|9=176|35=D|49=CLIENT12|56=BLOOMBERG|34=215|52=20230621-15:35:00.000|11=13579|21=1|48=US912828U816|22=1|54=1|60=20230621-15:34:56.000|38=1000|40=2|44=99.25|59=0|167=CORP|15=USD|10=145|
```

where

* `ClOrdID` (Tag 11): 13579 (Unique identifier for the order)
* `HandlInst` (Tag 21): 1 (Automated execution order, private)
* `SecurityID` (Tag 48): US912828U816 (ISIN for the bond)
* `SecurityIDSource` (Tag 22): 1 (CUSIP, Bloomberg uses 1 for ISIN as well)
* `Side` (Tag 54): 1 (Buy)
* `TransactTime` (Tag 60): 20230621-15:34:56.000 (Time the order was submitted)
* `OrderQty` (Tag 38): 1000 (Quantity of the order, typically in units of 1000 for bonds)
* `OrdType` (Tag 40): 2 (Limit order)
* `Price` (Tag 44): 99.25 (Price per bond)
* `TimeInForce` (Tag 59): 0 (Day order)
* `SecurityType` (Tag 167): CORP (Corporate bond)
* `Currency` (Tag 15): USD (Currency of the bond)

### C++ SDK: quickfix

