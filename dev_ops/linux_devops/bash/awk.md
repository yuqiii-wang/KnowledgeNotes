# AWK tool

`awk` provides a scripting language for text processing, of a given syntax `aws {cmd_scripts} <filename>`. Var denoted by `$` indicates str separated by `empty space`; `$0` represents the whole input str.

```bash
echo "Hello Tom" | awk '{$2="Adam"; print $0}'

# put awk_cmd in a file and run it
awk -f mypgoram.awk input.txt
```

## Loop every line from a file

`awk` follows `BEGIN{...}{...}END{...}` format in which the middle `{...}` processes each line of a file, given a delimiter specified by `FS`

Given a csv file such as
```bash
cat <<EOF>> data.csv
a,2
b,3
c,4
b,7
EOF
```

We can compute the summed b's value
```bash
awk 'BEGIN{
    FS=OFS=",";
    total_b = 0;
}
{
    key = $1;
    val = $2;
    if (key == "b")
        total_b += val;
}
END {
    printf "Total b is %s\n", total_b;
}' data.csv
```