-- Write a SQL query to find all numbers that appear at least three times consecutively.
SELECT DISTINCT newTbl.num AS ConsecutiveNums  FROM (
SELECT num, LEAD(num) over (ORDER BY id) AS num1, 
LEAD(num,2) over (ORDER BY id) AS num2
FROM Logs ) AS newTbl
WHERE newTbl.num = newTbl.num1 AND newTbl.num= newTbl.num2
-- LEAD is a sql server function that return the offset value