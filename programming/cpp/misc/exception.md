# Exception handling

Every function has a *Stack Frame* that pushes data into its stack for the purposes of
* function argument passing to the head of a stack
* return value
* store all existing local variables

A function that can properly catches an error when
* used `throw`
* used `try ... catch ...`

When an exception is thrown from a function, a good handling of an exception should be
* catch error info
* finish object lifecycle management

### Stack unwind

*stack unwind* is a concept in exception handling that a function on caught exception can successfully return with the two aforementioned conditions.

## Materialization

There are additional instruction insertions into compiled code, maintaining exception-related information tables. 

* `tblUnwind[]` is used to record objects to be released 

|Index|nextIndex|pfnDestroyer|pObj|
|-|-|-|-|
|0|1|`MyClass::~Var1()`|`&myClass`|
|1|2|`MyClass::~Var2()`|`&myClass`|
|2|3|`MyClass::~MyClass()`|`&myClass`|
|||...||

* `tblTryBlocks[]` is to record try code block information, identifying which line of code is wrong and the associated exception addr.

|Index|nextIndex|nEndSteps|tblCatchBlocksPtr|
|-|-|-|-|
|0|1|4|`&typeid(exception1)`|
|1|2|3|`&typeid(exception2)`|
|||...||

Comments: `nEndSteps` refers to the instruction offset relative to the start of a try block. 

* `tblCatchBlocks[]` records catch code block information.

|CatchBlockEntryPtr|piType|
|-|-|
|exception 1 pointer/addr|`&typeid(exception1)`|
|exception 2 pointer/addr|`&typeid(exception2)`|
||...|

## Cost analysis

The cost mainly relates to the size of the exception-related information tables and the inside destructors, plus the depth of stacks being pushed and popped out to return errors. 