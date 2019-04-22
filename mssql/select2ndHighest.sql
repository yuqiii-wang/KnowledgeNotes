/* T-SQL for select the second highest value of a column from a Employee table
references: https://leetcode.com/problems/second-highest-salary/submissions/ */
SELECT DISTINCT 
Salary AS SecondHighestSalary
FROM Employee
ORDER BY Salary DESC
OFFSET 1 ROWS
FETCH NEXT 1 ROWS ONLY