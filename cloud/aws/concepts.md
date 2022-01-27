# AWS Concepts

## AWS Systems Manager

AWS Systems Manager gives you visibility and control of your infrastructure on AWS.

* AWS Systems Manager Patch Manager

AWS Systems Manager Patch Manager automates the process of patching managed instances with both security related and other types of updates.

For example, a patching rule can be grouping instances by tags and choose a maintenace window (a time period) then select approved patch sources.

* Resource Manager

You can use resource groups to organize your AWS resources. Resource groups make it easier to manage and automate tasks on large numbers of resources at one time. 

For example, it can group resources (e.g., EC2 and S3) by tags.

* Parameter Store

AWS Systems Manager Parameter Store provides secure, hierarchical storage for configuration data management and secrets management. You can store data such as passwords, database strings, Amazon Machine Image (AMI) IDs, and license codes as parameter values. 

## AWS OpsWorks

AWS OpsWorks Stacks is a configuration management service that helps you build and operate highly dynamic applications and propagate changes instantly.

A Stack represents a set of instances that you want to manage collectively, typically because they have a common purpose such as serving PHP applications.

Chef Automate and Puppet Enterprise are DevOps tools similar to Ansible.

## Amazon Kinesis

Amazon Kinesis offers key capabilities to cost-effectively process streaming data at any scale.

* Amazon Kinesis Data Firehose 

Amazon Kinesis Data Firehose is a fully managed service for delivering real-time streaming data to destinations such as Amazon Simple Storage Service (Amazon S3), Amazon Redshift, Amazon Elasticsearch Service (Amazon ES), 

* Amazon Kinesis Data Streams

Kinesis Agent is a stand-alone Java software application that offers an easy way to collect and send data to Kinesis Data Streams.

For example, it can locally read Apache logs and translate into json format by records.

* Diff Between Data Streams and Firehose

One, Firehose is fully managed (i.e. scales automatically) whereas Streams is manually managed. Second, Firehose only goes to S3 or RedShift, whereas Streams can go to other services. Kinesis Streams on the other hand can store the data for up to 7 days.

## AWS Config

AWS Config is a service that enables you to assess, audit, and evaluate the configurations of your AWS resources. Config continuously monitors and records your AWS resource configurations and allows you to automate the evaluation of recorded configurations against desired configurations. 

Examples are statistics showing existing configurations compared against best practices for IAM.

## AWS Service Catalog

AWS Service Catalog enables organizations to create and manage catalogs of IT services that are approved for AWS. These IT services can include everything from virtual machine images.

Essentially, users of this offering write json or yaml config files listing required services they want to launch. Next time they can just run the config files.

## Amazon Inspector

Amazon Inspector is an automated security assessment service that helps improve the security and compliance of applications deployed on AWS, such as performing penetration attacks and scanning ports for vulnerabilities.

## AWS QuickInsight

A data visualization tool, similar to Power BI or Tableau.

## AWS Step Functions

Define the steps of your workflow in the JSON-based Amazon States Language. The visual console automatically graphs each step in the order of execution. Steps can be aws services such as Lambda.

## AWS Auto Scaling

It lets you choose scaling strategies to define how to optimize your resource utilization. For example, you can define a strategy that auto launches extra EC2 instances when existing instances' CPU usage reach 75%.

## Simple Queue Service (SQS)

Amazon Simple Queue Service (SQS) is a fully managed message queuing service that enables you to decouple and scale microservices, distributed systems, and serverless applications. 

For example, it can subscribe an CloudWatch Log and trigger a Lambda.

## AWS Elastic Beanstalk

You simply upload your code and Elastic Beanstalk automatically handles the deployment, from capacity provisioning, load balancing, and automatic scaling to web application health monitoring, with ongoing fully managed patch and security updates. 

## Amazon CloudFront

Amazon CloudFront is a fast content delivery network (CDN) service that securely delivers data, videos, applications, and APIs to customers globally with low latency. The diff from Route 53 is that it serves customers by best effort edge locations while Route 53 is by IP routing.

## RDS, Redshift, DynamoDB, and Aurora

**Redshift** is an OLAP database, standing for ‘online analytical processing’. This means it’s especially suited to processing analytical queries involving more complex calculations. Because of its vast storage potential and differing functionality, Redshift is sometimes referred to as a data warehouse.
Redshift is an enterprise-level DB, mostly used by large companies.

**DynamoDB** is a key-value database that runs the NoSQL engine. This makes the data stored in DynamoDB ‘dynamic’ which means it’s easier to modify. Key-value databases are best suited to certain use cases such as session data and shopping cart information and can achieve a fast throughput of read/write requests.

**Aurora** has significantly higher performance stats compared to MySQL and PostgreSQL run on RDS, and is MySQL and PostgreSQL compatible which means that the language which programs it is functionally similar to these engines, although it is an engine in its own right.

## AWS CloudWatch

* An **event** indicates a change in your AWS environment. AWS resources can generate events when their state changes. For example, Amazon EC2 generates an event when the state of an EC2 instance changes from pending to running, and Amazon EC2 Auto Scaling generates events when it launches or terminates instances.

* A **rule** matches incoming events and routes them to targets for processing. 

* A **target** processes events. Targets can include Amazon EC2 instances, AWS Lambda functions, Kinesis streams

## AWS Trusted Advisor

AWS Trusted Advisor is an online tool that provides you real time guidance to help you provision your resources following AWS best practices. Trusted Advisor checks help optimize your AWS infrastructure, increase security and performance, reduce your overall costs, and monitor service limits. 

## Elastic Beanstalk vs CLoud Formation

Elastic Beanstalk is a PaaS-like layer ontop of AWS's IaaS services which abstracts away the underlying EC2 instances, Elastic Load Balancers, auto scaling groups, etc. This makes it a lot easier for developers, who don't want to be dealing with all the systems stuff, to get their application quickly deployed on AWS. 

CloudFormation, on the other hand, doesn't automatically do anything. It's simply a way to define all the resources needed for deployment in a huge JSON file. So a CloudFormation template might actually create two ElasticBeanstalk environments (production and staging), a couple of ElasticCache clusters, a DyanmoDB table, and then the proper DNS in Route53. Since it's just a plain-text JSON file, I can stick it in my source control which provides a great way to version my application deployments. It also ensures that I have a repeatable, "known good" configuration that I can quickly deploy in a different region.

## Elastic Container Service (Amazon ECS)

**Amazon Elastic Container Registry (Amazon ECR)** is an AWS managed container image registry service.

**Amazon ECS cluster** is a logical grouping of tasks or services. If you are running tasks or services that use the EC2 launch type, a cluster is also a grouping of container instances.

Images that use the v2 or later format have a content-addressable identifier called a **digest**. As long as the input used to generate the image is unchanged, the digest value is predictable. `docker images --digests` to see image's digest.

**Amazon Machine Images (AMI)** 
provides the information required to launch an instance.

## AWS Cognito

Used to help user signin, signup, and social identity verification/access management service integration with 3rd party such as Google and Facebook.

For example, by React Native, Auth connects to backend servers.
```js
Amplify.configure({
    Auth: {
        identityPoolId: 'XX-XXXX-X:XXXXXXXX-XXXX', // Amazon Cognito Identity Pool ID
        region: 'XX-XXXX-X', // Amazon Cognito Region
    }
});
```

## AWS VPC

* Diffs between Interface and Gateway

![AwsVpcInterfaceVsGateway](../imgs/AwsVpcInterfaceVsGateway.png "AwsVpcInterfaceVsGateway")

* Prefix List

A prefix list is a set of one or more CIDR blocks, convenient to manage multiple CIDRs of similar purposes.