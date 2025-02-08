# Mac OS

## Homebrew

https://mirrors.tuna.tsinghua.edu.cn/help/homebrew/

### Tsinghua Mirror

```sh
export HOMEBREW_CORE_GIT_REMOTE="https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/homebrew-core.git"
```

## MacOS Commands

* `spctl`

`spctl` is a command-line tool on macOS used to interact with the System Policy control list.

`sudo spctl --master-enable`: This command enables the Gatekeeper settings that you configure through System Preferences > Security & Privacy > General.

`sudo spctl --master-disable` does the opposite.

Use case scenario:

For example, one use `sudo spctl --master-disable` to remove the protection, then re-enable it.

```txt
"jdk" can't be opened because Apple cannot check it for malicious software.
```
