# Jekyll and Github Pages

Jekyll is a static site generator with built-in support for GitHub Pages (the static website hosting service provided by Github `xxx.github.io`) and a simplified build process (just some yaml). Jekyll takes Markdown and HTML files and creates a complete static website. 

## Start

Install Ruby and check its version.
```
sudo apt-get install ruby-full ruby-bundler
ruby -v
```

To publish a user site, you must create a repository owned by your personal account that's named `<username>.github.io`

## Config

After the created repo with the name `<username>.github.io`, add `_config.yml`
```yml
lsi: false
safe: true
source: [your repo's top level directory, such as "/"]
incremental: false
highlighter: rouge
gist:
  noscript: false
kramdown:
  math_engine: mathjax
  syntax_highlighter: rouge
```
By default, Jekyll doesn't build files or folders that:
* are located in a folder called /node_modules or /vendor
* start with _, ., or #
* end with ~
* are excluded by the exclude setting in your configuration file