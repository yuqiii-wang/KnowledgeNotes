# provider helps install aws plugins
provider "aws" {
    region = "us-east-1"
    access_key = "xxx"
    secrete_key = "xxx"
}

resource "aws_instance" "my_instance" {
    ami = var.my_ami
    instance_type = "t2.micro"
    tags = var.my_ec2_tag
}

resource "aws_vpc" "my_vpc" {
    cidr_block = "10.0.0.0/16"
    tags = {
      Name = "Prod"
    }
}

resource "aws_subnet" "my_vpc_sub1" {
    vpc_id = aws_vpc.my_vpc.vpc_id
    cidr_block = "10.0.0.0/24"
    tags = {
      Name = "Prod-sub1"
    }
}

module "submodule_ec2_type" {
  source = "child_module"
}