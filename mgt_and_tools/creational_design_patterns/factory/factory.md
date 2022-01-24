# Factory

It's used when there's needs to manufacture many instances of different classes given configurations/parameters at run time.

```cpp
class Person {
private:
	std::string name;
	int age;
public:
	Person();
	~Person();
	void setName(std::string& name) {this->name = name;}
	int getAge() {return this->age;}
	void setAge(int age) {this->age = age;}
	std::string getName() {return this->name;}
};

class Student : public Person {
private:
	int grade;
public:
	Student();
	~Student();
	void setGrade(int grade) {this->grade = grade;}
	int getGrade() {return this->grade;}
};

class Professor : public Person {
private:
	int numPapers;
public:
	Professor();
	~Professor();
	void setNumPapers(int numPapers) {this->numPapers = numPapers;}
	int getNumPapers() {return this->numPapers;}
};

class PersonFactory {
public:
    static Person* newPerson(std::string& personRole) {
        if (personRole == "Professor") {
            return new Professor;
        }
        else if (personRole == "Student") {
            return new Student;
        }
        return nullptr;
    }
};
```