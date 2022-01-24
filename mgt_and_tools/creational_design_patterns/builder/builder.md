# Builder

When a described object is complex and needs lots of code inside a `constructor`, it is preferred to abstract common attributes and place them in a parent abstract class, and to realize by children with implementation details.

Class attributes are set/get methods, and ClassBuilder describes detailed/complex methods to set/get class attributes.

### Example

reference: https://en.wikibooks.org/wiki/C%2B%2B_Programming/Code/Design_Patterns#Builder


```cpp
class PersonMother {
public:
	void giveBirthPerson(PersonBuilder* pb) {
		personBuilder = pb;
		pb->setPerson();
	}
private:
	PersonBuilder* personBuilder;
}

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

// "Abstract Builder"
class PersonBuilder {
public:
	virtual ~PersonBuilder() {};

	Person* getPerson() {
		return person.get();
	}
	void setPerson(){
		person = std::make_unique<Person>();
	}
	virtual void buildProfessor(int numPapers) = 0;
	virtual void buildStudent(int grade) = 0;
protected:
	std::unique_ptr<Person> person;
};

class ProfessorBuiler : public PersonBuilder {
public:
	void buildProfessor(std::string name, int age, int numPapers) {
		person->setName(name);
		person->setNumPapers(numPapers);
		person->setAge(age);
	}
};

class StudentBuiler : public PersonBuilder {
public:
	void buildStudent(std::string name, int age, int grade) {
		person->setName(name);
		person->setGrade(grade);
		person->setAge(age);
	}
};

int main(){
	PersonMother pm;
	StudentBuiler sb("SB", 18, 90);
	ProfessorBuiler pb("PB", 45, 10);

	pm.giveBirthPerson(&sb);
	pm.giveBirthPerson(&pb);

	return 0;
}
```