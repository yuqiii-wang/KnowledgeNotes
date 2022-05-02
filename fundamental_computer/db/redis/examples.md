# Redis Example Uses

## Autocomplete

In the web world, autocomplete is a method that allows us to quickly look up things that
we want to find without searching. Generally, it works by taking the letters that we’re
typing and finding all words that start with those letters.

* Solution:

Use `LIST` to push/pop to add/remove recent searched words; Each item is a `String` so that substring compare can be used to match prefix.

## Cloud distributed locks

In Cloud app development, services are micro modules interacting with each other. There should be locks to prevent service-level concurrency.

* Solution:

Use `String` to set bool value to indicate if there is a lock.