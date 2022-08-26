resource "aws_instance" "my_instance_nano" {
    ami = var.my_ami
    instance_type = var.example_ec2_nano
}
