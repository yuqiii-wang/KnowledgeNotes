# Some Development Tips

* It's a declarative language, so that, for example, when you declare two resources with same configs in a same file, it would **only deploy one** (all deployments are cached). You need **for_each** to explicitly inform terraform to create multiple recources.

* **.terraform** folder saves plugins, **terraform.tfstate** tracks history implementations.

* **backend** can be used for sync deployment.
```json
terraform {
    backend "s3" {
        bucket = "xxx"
        key = "xxx"
        region = "us-east-1"
    }
}
```

* **.tfvars** can inform terraform to auto load variables declared in files with such a suffix. Variables and tfvars are of a global scope.

* Variable Definition Precedence

Terraform loads variables in the following order, with later sources taking precedence over earlier ones:

1. Environment variables
2. The terraform.tfvars file, if present.
3. The terraform.tfvars.json file, if present.
4. Any *.auto.tfvars or *.auto.tfvars.json files, processed in lexical order of their filenames.
5. Any -var and -var-file options on the command line, in the order they are provided. (This includes variables set by a Terraform Cloud workspace.)