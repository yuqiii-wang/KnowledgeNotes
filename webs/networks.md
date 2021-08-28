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