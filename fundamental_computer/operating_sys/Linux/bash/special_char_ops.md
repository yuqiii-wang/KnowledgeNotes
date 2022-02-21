# Special Character Operations

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

`$#`, `$@` and `$?`:
Run the bash script
```bash
#! /bin/sh
echo '$#' $#
echo '$@' $@
echo '$?' $?
```
You get output:

$#  3,  number of arguments. Answer is 3

$@  1 2 3, what parameters were passed. Answer is 1 2 3

$?  0, was last command successful. Answer is 0 which means 'yes'
