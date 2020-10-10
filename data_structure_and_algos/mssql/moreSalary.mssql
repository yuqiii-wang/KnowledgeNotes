-- Given the Employee table, write a SQL query that finds out employees who earn more than their managers. 
SELECT
    table_a.Name AS Employee
FROM
    Employee AS table_a,
    Employee AS table_b
WHERE
    table_a.ManagerId = table_b.Id
        AND table_a.Salary > table_b.Salary
;