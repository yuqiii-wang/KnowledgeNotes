# Git Commands

##How to upload
First, go to `https://github.com/settings/tokens/new` and generate token with `write:packages` permission. Then put that token onto the `<token>` in the bash below.

```bash
#set token
username=yuqiii-wang
token=<token>

# reset remote origin
git remote set-url origin https://${username}:${token}@github.com/yuqiii-wang/InterviewQuesPractices.git/

```

