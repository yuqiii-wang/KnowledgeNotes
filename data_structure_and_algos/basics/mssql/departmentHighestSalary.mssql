-- Write a SQL query to find employees who have the highest salary in each of the departments.
/* Write your T-SQL query statement below */
SELECT Department.Name AS Department,
    Employee.Name AS Employee,
    Employee.Salary AS Salary
FROM Employee 
    JOIN Department 
    ON Employee.DepartmentId=Department.Id
    JOIN (SELECT DepartmentId, MAX(Salary)
            FROM Employee
            GROUP BY DepartmentId
        ) AS X (DepartmentId, Salary) 
    ON Employee.DepartmentId=X.DepartmentId 
        AND Employee.Salary=X.Salary