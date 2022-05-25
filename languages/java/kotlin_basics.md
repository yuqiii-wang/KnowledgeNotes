# Kotlin Basics

* `companion object`

Same as below, works like a static function without instantiating an object

```kotlin
class ToBeCalled {
    fun callMe() = println("You are calling me :)")
}
fun main(args: Array<String>) {     
    val obj = ToBeCalled()
    
    // calling callMe() method using object obj
    obj.callMe()
}
```