#include <iostream>

/*
  In CRTP, basic attrs are defined in base class, and their implementations are defined
  in derived classes.
*/

template<typename T>
class Person
{
public:
  std::string getName() const { return static_cast<T*>(this)->getName(); }
  int getAge() const { return static_cast<T*>(this)->getAge(); }

  void setName(std::string _name) const { static_cast<T*>(this)->setName(_name); }
  void setName(std::string& _name) const { static_cast<T*>(this)->setName(_name); }
  void setAge(int _age) const { static_cast<T*>(this)->setAge(_age); }
};

class Professor : Person<Professor>
{
public:
  std::string getName() { return name; }
  int getAge() { return age; }
  int getSalary() { return salary; }

  void setName(std::string _name) { name=_name; }
  void setName(std::string& _name) { name=_name; }
  void setAge(int _age) { age=_age; }
  void setSalary(int _salary) { salary=_salary; }
private:
  std::string name;
  int age;
  int salary;
};

class Student : Person<Professor>
{
public:
  std::string getName() { return name; }
  int getAge() { return age; }
  int getGrade() { return grade; }

  void setName(std::string& _name) { name=_name; }
  void setName(std::string _name) { name=_name; }
  void setAge(int _age) { age=_age; }
  void setGrade(int _grade) { grade=_grade; }
private:
  std::string name;
  int age;
  int grade;
};

int main()
{
  Student s1;
  Professor p1;

  s1.setName(std::string{"Dick"});
  s1.setAge(18);
  s1.setGrade(80);

  p1.setName(std::string{"Yuqi"});
  p1.setAge(29);
  p1.setSalary(80000);

  std::cout << p1.getName() << " age: " << p1.getAge() << ", salary: " << p1.getSalary() << std::endl;
  std::cout << s1.getName() << " age: " << s1.getAge() << ", grade: " << s1.getGrade() << std::endl;

  return 0;
}