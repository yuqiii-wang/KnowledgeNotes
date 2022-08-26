# Prototype

New instances are determined/produced by a prototypical object with `clone` method. The cloned instances have shared attributes (shallow copy of the prototype).

It is useful when `constructor` is prohibitively expensive (for example, many likely non-changing attributes be initialized when calling a `constructor`), and instantiated objects with attributes of the same shared memory can 

```cpp
class Person {
private:
	std::string name;
	int age;

    struct ManyHumanAttributes {
        bool isBloodRed = true;
        bool isUsuallyHavingHair = true;
        bool isUsuallyEitherMaleOrFemale = true;
        bool isUsuallyHavingTenFingers = true;
        bool isUsuallyHavingTentoes = true;
        bool isUsuallyHavingTwoEyes = true;
        bool isUsuallyHavingOneNose = true;
        bool isUsuallyHavingOneMouth = true;
        bool isUsuallyHavingTwoEars = true;
        bool isUsuallyHavingTeeth = true;
        int monthPregancyBeforeDelivery = 10;
        int maxAge = 120;
    } manyHumanAttributes;

public:
	Person();
	~Person();
	void setName(std::string& name) {this->name = name;}
	int getAge() {return this->age;}
	void setAge(int age) {this->age = age;}
	std::string getName() {return this->name;}

    virtual std::unique_ptr<Person> clone() = 0;
};

class Student : public Person {
private:
	int grade;
public:
	Student();
	~Student();
	void setGrade(int grade) {this->grade = grade;}
	int getGrade() {return this->grade;}

    unique_ptr<Person> clone() override {
		return std::make_unique<Student>(*this);
	}
};

class Professor : public Person {
private:
	int numPapers;
public:
	Professor();
	~Professor();
	void setNumPapers(int numPapers) {this->numPapers = numPapers;}
	int getNumPapers() {return this->numPapers;}

    unique_ptr<Person> clone() override {
		return std::make_unique<Professor>(*this);
	}
};

class PersonFactory {
private:
    std::unordered_map<std::string, std::unique_ptr<Person>> person_dict;
public:
    // when PersonFactory is built, it finishes init shared memory attributes
    PersonFactory() {
        person_dict["Professor"] = std::make_unique<Person>();
        person_dict["Student"] = std::make_unique<Student>();
    }

    std::unique_ptr<Person> newPerson(std::string& personRole){
        return person_dict[personRole]->clone();
    }
};

int main(){
    PersonFactory personFactory;
    personFactory.newPerson("Student");
    personFactory.newPerson("Professor");

    return 0;
}
```