# Terraform

The main purpose of the Terraform language is declaring resources, which represent infrastructure objects. All other language features exist only to make the definition of resources more flexible and convenient.

For example, for aws management, terraform wraps aws cli, so that user can write json-like declarative scripts to manage aws infrastructure.s

In Terraform CLI, the **root module** is the working directory where Terraform is invoked. 

## Resource

A **resource** block declares a resource of a given type ("aws_instance") with a given local name ("web"). 

```json
resource "aws_instance" "web" {
  ami           = "ami-a1b2c3d4"
  instance_type = "t2.micro"
}
```

Each resource type is implemented by a **provider**, which is a plugin for Terraform that offers a collection of resource types. 

```json
provider "aws" {
    alias = "sg"
    region = "ap-southeast-1"
}
```

## Data

**Data** sources allow data to be fetched or computed for use elsewhere in Terraform configuration.

For example, find the latest available AMI that is tagged with Component = web
```json
data "aws_ami" "web" {
  filter {
    name   = "state"
    values = ["available"]
  }

  filter {
    name   = "tag:Component"
    values = ["web"]
  }

  most_recent = true
}
```

Then it can be used in resource (in aws, terraform retrieves (by running aws cli) key-value pairs according to the filter rules, for example, image_id (ec2 AMI id) is a return result among many others).
```json
resource "aws_instance" "web" {
  ami           = data.aws_ami.web.image_id
  instance_type = "t1.micro"
}
```

## Variables

**Input** variables serve as parameters for a Terraform module, allowing aspects of the module to be customized without altering the module's own source code, and allowing modules to be shared between different configurations.

```json
variable "user_information" {
  type = object({
    name    = string
    address = string
  })
  sensitive = true
}

resource "some_resource" "a" {
  name    = var.user_information.name
  address = var.user_information.address
}
```

Scope of a variable is by default constraint to the current dir.

**Output** values are like the return values of a Terraform module,
```json
output "instance_ip_addr" {
  value = aws_instance.server.private_ip
}
```

A **local** value assigns a name to an expression, so you can use it multiple times within a module without repeating it.
```json
locals {
  service_name = "forum"
  owner        = "Community Team"
}
```

## Modules

**Modules** are containers for multiple resources that are used together.

```json
module "servers" {
  source = "s3::https://s3-eu-west-1.amazonaws.com/examplecorp-terraform-modules/vpc.zip"
  servers = 5
}
```

**Source** value is either the path to a local directory containing the module's configuration files, or a remote module source that Terraform should download and use. Terraform uses this during the module installation step of `terraform init`

## Expressions

A ${ ... } sequence is an **Interpolation**
```json
"Hello, ${var.name}!"
```

A %{ ... } sequence is a **directive**
```json
"Hello, %{ if var.name != "" }${var.name}%{ else }unnamed%{ endif }!"
```

`data.<DATA TYPE>.<NAME>` is an object representing a data resource.

`module.<MODULE NAME>.<OUTPUT NAME>` is the value of the specified output value 

## State

Terraform must store **state** about your managed infrastructure and configuration. This state is used by Terraform to map real world resources to your configuration, keep track of metadata, and to improve performance for large infrastructures.