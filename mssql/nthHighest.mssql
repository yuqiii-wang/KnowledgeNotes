/* Write a T-SQL query to get the nth highest salary from the Employee table. */
CREATE FUNCTION getNthHighestSalary(@N INT) RETURNS INT AS
BEGIN
    RETURN (
        /* Write your T-SQL query statement below. */
        SELECT DISTINCT 
        Salary AS SecondHighestSalary
        FROM Employee
        ORDER BY Salary DESC
        OFFSET @N-1 ROWS
        FETCH NEXT 1 ROWS ONLY
    );
END