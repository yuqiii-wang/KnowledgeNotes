# Derivative

A type of financial contract whose value is dependent on an underlying asset, group of assets, or benchmark. 

The most common underlying assets for derivatives are stocks, bonds, commodities, currencies, interest rates, and market indexes. Contract values depend on changes in the prices of the underlying asset. 

Some common financial contracts are futures (an underlying asset traded on a specific date and contracts itself can be traded), options (right to buy/sell an underlying asset)

## Swap

A swap is a derivative contract through which two parties exchange the cash flows or liabilities from two different financial instruments. The two product to be swapped are not necessary the same underlying asset type.

For example, Party A regards the price of one security *SB* rising and currently holds another security *SA* expecting going down; Party B thinks of the opposite and currently holds *SB*. They can sign a swap contract for speculation/risk hedging.

### CDS (Credit Default Swap)

A credit default swap (CDS) is a financial swap agreement that the seller of the CDS will compensate the buyer in the event of a debt default (by the debtor) or other credit event.

In other words, it is an insurance to an event. In 2008 financial crisis, people buy CDS from banks to insure the event that some real estate companies would not bankrupt. People paid premiums to banks, and banks repaid a huge sum of money as compensation to people when those real estate companies saw credit defaults.

## Futures

Derivative financial contracts that obligate the parties to transact an asset at a predetermined future date and price. 

* Parties can `long`/`short` derivatives, a typical work flow such as

1) Given a spot price of $1,000 for one unit of a derivative contract at a time

2) Parties expecting a higher close price can buy (bought the derivative, spent $1,000, paid to the selling party, who received $1,000 per unit) at the spot price (called `long`), while the opposite expecting lower can sell (called `short`). Here assumes both buy/sell one unit of the derivative.

3) There is no actual purchase action on the commodity stipulated on the derivative contract, just two opposite parties make order to each other. The selling party does not need to possess (not already purchased) any unit of the derivative-defined commodity. The selling party is just labelled `-1` to its account book, accordingly, the buy side's account book receives `+1`.

4) As time goes by, the spot price of the derivative goes up and down. Parties can trade before the clearing date, or not to.

5) On the date of clearing, the spot goods (the derivative contract defined commodity) must be purchased/sold by the parties.

6) If the spot price of the clearing date is lower than $1,000, for example, $300. The selling party must prepare one unit of the actual derivative-defined commodity by spending $300 to purchase one, and the buying party must pay $300 to the selling party and receives the commodity.

7) The selling party buys the actual physical commodity from a third party (usually through an exchange broker), since the selling party does not reserve any physical commodity, just wants to speculate. The buying party once receives the physical commodity, can immediately sell it to another third party at the same spot price of $300, since it has no interest in preserving the commodity, just wants to speculate as well. The exchange of commodity does not even happen sometimes, just two parties speculating against each other.

8) Hence, the selling party (the short) earned $700 ($1,000 - $300) as profit, while the buying party (the long) lost $700 accordingly. The logic holds truth if the clearing date price went up, that indicates loss for the short and win for the long.

## Options

An options contract offers the buyer the opportunity to buy or sell, depending on the type of contract they hold—the underlying asset. There is no obligation for contract participants to exercise buy/sell.

Call options allow the holder to buy the asset at a stated price within a specific timeframe. Put options, on the other hand, allow the holder to sell the asset at a stated price within a specific timeframe.

Example: Daily Oil investment, where people pays a premium to obtain a right to buy/sell oil at a stated price within a day.