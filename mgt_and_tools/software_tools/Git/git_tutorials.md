# Git Commands

## git clone with token

```bash
git clone https://github.com/username/repo.git
```
input on prompts: **password is token instead**
```bash
Username: <username>
Password: <token>
```


## How to upload
First, go to `https://github.com/settings/tokens/new` and generate token with `write:packages` permission. Then put that token onto the `<token>` in the bash below.

```bash
#set token
username=<username>
token=<token>
reponame=<reponame>

# reset remote origin
git remote set-url origin https://${username}:${token}@github.com/${username}/${reponame}.git

```

## Resolve conflicts

* `git pull`

This happens after `git pull` and finished selecting code merge
```bash
git commit -am "your commit msg"
```

Use this for `git rebase <new_branch>`
```bash
git rebase --continue
```

* `git switch` (starting from v2.23)

Happens when using `git switch`, run `git switch <branch_name> -f` to ingonre changes and force switch to another branch. 

## git rollback

* rollback from `git add .`

```bash
git reset .
```

* rollback from `git commit`

```bash
git log --oneline
# find HEAD hash
git reset <HEAD>
```

## git delete branch

```bash
# locally
git branch -d <localbranchname>

# remote
git push origin --delete <remotebranchname>
```