# System Manager

## Association

A State Manager association is a configuration that you assign to your AWS resources. The configuration defines the state that you want to maintain on your resources. For example, an association can specify that antivirus software must be installed and running on a managed node, or that certain ports must be closed.

Below is the code that create, launch and delete an association that allows aws to auto update ssm agent
```bash
ec2Id=$(curl http://169.254.169.254/latest/meta-data/tags/instance)
region=$(curl http://169.254.169.254/latest/meta-data/tags/placement/availability-zone)

aws ssm create-associations --instance-id ${ec2Id} --name "AWS-UpdateSSMAgent" --region ${region} > assoResult.txt
asso=$(cat assoResult.txt | grep AssociationId | sed "s/\,//g")

IFS=' ' # shell flag: internal field separator
read -ra assoId <<< "${asso}"
assoId=$(echo $${assoId[1]} | sed "s/\"//g")
awsa ssm start-associations-once --association-ids ${assoId} --region ${region}
sleep 10
aws ssm delete-association --instance-id ${ec2Id} --name "AWS-UpdateSSMAgent" --region ${region}
```