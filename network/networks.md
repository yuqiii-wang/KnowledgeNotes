# Network Knowledge

## Network Time Protocol

Allow computers to access network unified time to get time synced. 

Stratum Levels:
* Stratum 0: Atomic clocks
* Stratum 1 - 5: various Time Servers
* Stratum 16: unsynced

Time authentication fails for large time gaps.

Primary NTP servers provide first source time data to secondary servers and forward to other NTP servers, and NTP clients request for time sync info from these NTP servers.

## Kerberos

A computer-network authentication protocol that works on the basis of tickets to allow communication nodes communicating over a non-secure network to prove their identity to one another in a secure manner.


## MIME

Multipurpose Internet Mail Extensions (MIME) is an Internet standard that extends the format of email messages to support text in character sets other than ASCII, as well as attachments of audio, video, images, and application programs. 

Inside a request's header, `Content-Type` specifies media type, such as
```bash
Content-Type: text/plain
Content-Type: application/json
Content-Type: application/octet-stream
Content-Type: application/x-www-form-urlencoded
```