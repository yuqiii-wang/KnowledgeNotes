# Linux Bash Common Interview Questions and Commands

## Cheat sheets： 

https://devhints.io/bash

https://mywiki.wooledge.org/BashSheet#:~:text=An%20ampersand%20does%20the%20same,for%20the%20former%20to%20end.

##  Questions:

1. find a keyword contained in a specified file name

```bash
find -name "<filename>" | xargs grep <keyword>
```

2. echo "Hello World" for every 2 second

```bash
while true
do 
    echo "Hello World"
    sleep 2
done

# or equivalent in one liner
while sleep 2; do echo "Hello World"; done
```
3. write a function that takes one argument and return a result

```bash
function myfunc() {
    echo "hello $1"
}
result=$(myfunc "John")
echo $result
```

4. how to test an if condition

```bash
# test a string
string=hello
if [[ -z "$string" ]]; then
  echo "String is empty"
elif [[ -n "$string" ]]; then
  echo "String is not empty, it is $string"
else
  echo "This never happens"
fi

# test a file
if [[ -e "file.txt" ]]; then
    echo "file exists"
else
    echo "file does not exist"
fi
```
condition reference:
![alt text](imgs/bash_conditions.png "bash_conditions")

5. ./ vs source

    `./script` runs the script as an executable file, launching a new shell to run it.

    `source script` reads and executes commands from filename in the current shell environment

    *Note*: `./script` is not `. script`, but `. script` == `source script`

6. current shell vs sub shell

Whenever you run a shell script, it creates a new process called subshell and your script will get executed using a subshell. If you start another shell on top of your current shell, it can be referred to as a subshell.
```bash
# to see whether this shell is a parent shell
echo $BASH_SUBSHELL

# a sub shell does not inherit variable env, use export to sync var
# export: Export the given variable to the environment so that child processes inherit it.
var=1
export var
```

7. processes in shell

`jobs`: List the current shell's active jobs.

`bg`: Send the previous job (or job denoted by the given argument) to run in the background.

`fg`: Send the previous job (or job denoted by the given argument) to run in the foreground.

```bash
# The exec command replaces this shell with the specified program without swapping a new subshell or proces (after execution, the shell exits)
exec echo "hello"
```
About running processes in background
```bash
# In Linux, /dev/null is a special device file which writes-off (gets rid of) all data written to it, in the command above, input is read from, and output is sent to /dev/null.
# For example:

grep -r hello /sys/ 2> /dev/null # here /dev/null disables many Permission denied std err output
```
Use nohup command, which also enables a process to continue running in the background when a user exits a shell。
```bash
nohup cmd & exit
nohup nice cmd & exit # nice makes the cmd low priority to run
```

Alternatively, a trailing `&` can run a command in background, such as
```bash
echo $! # returns a proc id

jobs # use jobs to see currently running jobs in this shell

fg # this job can be brought to front (if not yet finished/done)
```

8. special character operators

`[word] [space] [word]`
Spaces separate words. In bash, a word is a *group of characters* that belongs together.

`'[Single quoted string]'`
*Disables syntactical meaning* of all characters inside the string. 

`"[Double quoted string]"`
Disables syntactical meaning of all characters *except expansions inside the string*. Use this form instead of single quotes if you need to expand a parameter or command substitution into your string.

`[command] ; [command] [newline]`
Semi-colons and newlines *separate synchronous commands* from each other. Use a semi-colon or a new line to end a command and begin a new one. The first command will be executed synchronously, which means that Bash will wait for it to end before running the next command.

`[command] & [command]`
A single ampersand *terminates an asynchronous command*. An ampersand does the same thing as a semicolon or newline in that it indicates the end of a command, but it causes Bash to execute the command asynchronously. That means Bash will run it in the background and run the next command immediately after, without waiting for the former to end. Only the command before the & is executed asynchronously and you must not put a ; after the &, the & replaces the `;`.

`[command] | [command]`
A vertical line or pipe-symbol *connects the output of one command to the input of the next*. Any characters streamed by the first command on stdout will be readable by the second command on stdin.

`[command] && [command]`
An AND conditional causes the second command to be executed *only if the first command ends* and exits successfully.

`[command] || [command]`
An OR conditional causes the second command to be executed *only if the first command ends and exits with a failure exit code* (any non-zero exit code).

`"$var", "${var}"`
Expand the value contained within the parameter var. The parameter expansion syntax is replaced by the contents of the variable.

`(([arithmetic expression]))`
Evaluates the given expression in an *arithmetic context*. For example:
```bash
echo $((1+5))
printf %.2f\\n "$((10**3 * 2/3))e-3"
```

`{[command list];}`
Execute the list of commands in the current shell as though they were one command. It would be useful such as
```bash
rm filename || { echo "Removal failed, aborting."; exit 1; }
```

`([command list])`
*Execute the list of commands in a subshell*.
This is exactly the same thing as the command grouping above, only, the commands are executed in a subshell. Any code that affects the environment such as variable assignments, cd, export, etc. do not affect the main script's environment but are scoped within the brackets.

`[command] > [file], [command] [n]> [file], [command] 2> [file]` 
File Redirection: The `>` operator redirects the command's *Standard Output (or FD n) to a given file*. The number indicates which file descriptor of the process to redirect output from. The file will be truncated *(emptied)* before the command is started!

`[command] >> [file], [command] [n]>> [file]`
File Redirection: The >> operator redirects the command's Standard Output to a given file, *appending* to it.

`[command] <([command list])`
Process substitution: The `<(...)` operator expands into a new file created by bash that contains the other command's output.

`2>&1` stderr redirected to stdout. 2 is interpreted as file descriptor 2 (aka stderr) and file descriptor 1 as stdout in the context of stream (P.S. 0 for stdin).
```bash
# the two different cmds give different colours of output
g++ lots_of_errors 2>&1 | head
g++ lots_of_errors 2>&2 | head
```

`[command] "$([command list])"`
Command Substitution: captures the output of a command and expands it inline.

`? * [...]` Glob (regex) indicators: common regex syntax applies here.

`[command] &` This trailing ampersand directs the shell to run the command in the background, that is, it is forked and run in a separate sub-shell, as a job, asynchronously.