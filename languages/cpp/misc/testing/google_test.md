# GoogleTest

## `GTest::Main`

By using `GTest::Main` in `target_link_libraries`, code automatically loads the `main` function so that programmer does not need to explicitly write the `main` function.
```bash
target_link_libraries(googletest PUBLIC 
                        GTest::GTest
                        GTest::Main)
```

The `main` function is listed as below.
```cpp
#if GTEST_OS_ESP8266 || GTEST_OS_ESP32

#if GTEST_OS_ESP8266
extern "C" {
#endif

void setup() { testing::InitGoogleTest(); }

void loop() { RUN_ALL_TESTS(); }

#if GTEST_OS_ESP8266
}
#endif

#elif GTEST_OS_QURT
// QuRT: program entry point is main, but argc/argv are unusable.

GTEST_API_ int main() {
  printf("Running main() from %s\n", __FILE__);
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}
#else
// Normal platforms: program entry point is main, argc/argv are initialized.

GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from %s\n", __FILE__);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
```

## `TEST`

`TEST(test_case_name, test_name)` test_case_name is the name of the Test Case. test_name is the name of the
individual scenario contained within this test method. A test case can and
probably should contain multiple tests. 

## Test Fixture `TEST_F`

Test fixture is a place to hold objects and functions shared by
all tests in a test case.  Using a test fixture avoids duplicating
the test code necessary to initialize and cleanup those common
objects for each test.  It is also useful for defining sub-routines
that your tests need to invoke a lot.

User class should inherit from `testing::Test`.

To be tested members should be in protected scope for derived class access

Virtual function `SetUp` and `TearDown` are used for init/destroy variables.

Also, instead of directly implementing assertions such as `ASSERT_EQ` inside class definition, tests can be defined in `TEST_F`.

```cpp
class YourClass : public testing::Test {
 protected:  // You should make the members protected s.t. they can be
             // accessed from sub-classes.

  // virtual void SetUp() will be called before each test is run.  You
  // should define it if you need to initialize the variables.
  // Otherwise, this can be skipped.
  void SetUp() override {
  }

  // virtual void TearDown() will be called after each test is run.
  // You should define it if there is cleanup work to do.  Otherwise,
  // you don't have to provide it.
  virtual void TearDown() {
  }

  // here can define and run `ASSERT_EQ`
  void runService(){
    ASSERT_EQ(0, 0);
  }
};

// Tests the default c'tor.
TEST_F(YourClass, DefaultConstructor) {
  // You can access data in the test fixture here.
  EXPECT_EQ(0u, 0u);
}
```

## Event Listener Testing

Event listener testing is used to trigger tests on received signals, such as  

To define a event listener, you subclass either `testing::TestEventListener` or `testing::EmptyTestEventListener`. The former is an (abstract) interface, where each pure virtual method can be overridden to handle a test event.

`OnTestStart` and `OnTestEnd` are virtual function to be overridden before and after testing accordingly. `OnTestPartResult` can show on-going testing results.

```cpp
class YourClass : public testing::EmptyTestEventListener
{
    // Called before a test starts.
    void OnTestStart(const testing::TestInfo& test_info) override {
      printf("*** Test %s.%s starting.\n",
             test_info.test_suite_name(), test_info.name());
    }

    // Called after a failed assertion or a SUCCESS().
    void OnTestPartResult(const testing::TestPartResult& test_part_result) override {
      printf("%s in %s:%d\n%s\n",
             test_part_result.failed() ? "*** Failure" : "Success",
             test_part_result.file_name(),
             test_part_result.line_number(),
             test_part_result.summary());
    }

    // Called after a test ends.
    void OnTestEnd(const testing::TestInfo& test_info) override {
      printf("*** Test %s.%s ending.\n",
             test_info.test_suite_name(), test_info.name());
    }
};
```