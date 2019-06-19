// Find valid IPv4 from a string

// regex explain:
// (?=$|[^\w.])
// (?:1?\d?\d|2[0-4]\d|25[0-5])
// () capturing a group of chars; ?: do not back referencing;
// ? indicates zero or one repetition;
// 1?\d?\d any one digit | any two digits | three digits start with one;

var regexStr = /(?:1?\d?\d|2[0-4]\d|25[0-5])(?:\.(?:1?\d?\d|2[0-4]\d|25[0-5])){3}/g

var testStr = `.1e2- 9yp8aoiw roqwepru9p23n<hello>
sosihf sfljq=3r0 8, 123.34.2.5, fsdk ewsdn333,222,.2321
123.444.23.4, 23,2,3,124.123.123.123.4.3a`

console.log(regexStr.exec(testStr));
console.log(regexStr.exec(testStr));
console.log(regexStr.exec(testStr));