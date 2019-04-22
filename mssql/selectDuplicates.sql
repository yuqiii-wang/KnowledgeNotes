-- Write a SQL query to find all duplicate emails in a table named Person.

/* Write your T-SQL query statement below */

SELECT Email
FROM Person
GROUP BY Email
HAVING COUNT(Email) > 1; -- after group by, WHERE is not allowed, replaced with HAVING