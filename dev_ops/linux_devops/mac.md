# Mac OS

* Homebrew

`Homebrew` is the default mac software repository.

In China where github is blocked, use this.

```sh
# export HOMEBREW_API_DOMAIN=
export HOMEBREW_BREW_GIT_REMOTE="https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/brew.git"
brew update
brew install <package>
```

* find largest folder on Mac.

`du -h /System/Volumes/Data | grep "G\t" | sort`.