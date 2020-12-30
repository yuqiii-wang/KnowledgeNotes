variable "my_ami" {
    type = string
    description = "my simple ami"
}

variable "my_ec2_tag" {
    type = object({
        Name = string
        foo = string
    })
}