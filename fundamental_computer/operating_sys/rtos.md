# RTOS (Real Time Operating System)

RTOS (Real Time Operating System) is characterised by operations completed within a known, bounded timeframe.

||RTOS|Linux|
|-|-|-|
|Typical Use|Embedded systems, control systems, safety-critical|Desktops, servers, mobile devices, complex embedded|
|Complexity|Simpler, focused functionality, small OS kernel|Highly complex, rich feature set, large OS kernel|
|Development|Often bare-metal feel, specialized tools, fine-grained hardware control|Rich ecosystem, high-level tools, vast libraries, abstract hardware control|

## RTOS Implementations for Real-Time Operations

* Priority-Based Preemptive Scheduler

Process scheduler always ensures that the highest-priority task that is ready to run is the one currently executing.

* Minimal and Bounded Interrupt Latency

Interrupt latency is the time from when a hardware interrupt occurs to when the first instruction of the Interrupt Service Routine (ISR) begins executing.
RTOS has optimizations on awakening interrupts.

* Deterministic Synchronization Primitives (with Priority Inversion Avoidance)

Priority inversion occurs when a high-priority task is blocked waiting for a resource held (mutexes, semaphores, and message queues for tasks to synchronize and communicate) by a lower-priority task, which itself might be preempted by a medium-priority task.

RTOS implements interrupt priority re-arrangement to make sure the highest priority task get picked up, and once the highest priority task completes, the remaining tasks are unaffected.

* Predictable Memory Management

Many RTOSs avoid dynamic memory allocation (malloc(), free()) in time-critical paths or use highly deterministic memory allocation schemes.

Static Allocation: Memory for tasks, stacks, and objects is allocated at compile-time or system startup.

Fixed-Size Block Pools: Memory is divided into pools of fixed-size blocks, hence no fragmentation.

## Development Start Guide (Exampled by ESP32S3 Chip)

The code below is simply set an LED to blink.

```c
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include <stdio.h>
#include "led.h"


void app_main(void)
{
    esp_err_t ret;
    
    ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    led_init();

    while(1)
    {
        LED0_TOGGLE();
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}
```

where

* `LED0_TOGGLE()`: Toggles the state of LED0 (turns it on if it was off, and off if it was on).
* `pdMS_TO_TICKS(500)`: Converts 500 milliseconds into the equivalent number of RTOS "ticks" (the basic unit of time for the RTOS scheduler). The actual duration of a tick is configured by `configTICK_RATE_HZ`.
* `vTaskDelay()`: Puts the current task (the one executing app_main) into the Blocked state for the specified number of ticks.

### How `app_main` works

In summary, `app_main(void)` is registered as a task in RTOS.

In detail:

1. Power-On/Reset
2. 1st Stage Bootloader (ROM): Basic hardware initialization, loads 2nd stage bootloader.
3. 2nd Stage Bootloader (Flash):
    * More hardware initialization (e.g., SPI flash, clock configuration).
    * Loads the application image (which includes your code, ESP-IDF libraries, and FreeRTOS) into RAM.
4. ESP-IDF Startup Code (within the loaded application image):
    * Performs further system initialization (e.g., heap initialization, C library setup).
    * Initializes FreeRTOS
    * Creates the "Main Task" (or "Application Task") and specified
        * The function pointer provided to `xTaskCreate` for this new task is set to point to your `app_main` function.
        * stack is allocated for this task.
        * A priority is assigned (typically `tskIDLE_PRIORITY + 1` or a configurable application main task priority).
        * The task is added to the RTOS's list of ready tasks.
5. Starts the FreeRTOS Scheduler: The ESP-IDF startup code calls `vTaskStartScheduler()` by which `app_main(void)` is managed in RTOS