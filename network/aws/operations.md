# Best AWS Practices

## Use Inspector with EC2

To start using **Amazon Inspector with these EC2 instances**, tag them to match the assessment target that you want. The configuration of Amazon Linux 2 AMI with Amazon Inspector Agent enhances security by focusing on two main security goals: limiting access and reducing software vulnerabilities.

This is the only currently available EC2 instance AMI with the preinstalled Amazon Inspector agent. For the EC2 instances that run Ubuntu Server or Windows Server, you must complete the manual agent installation steps.

Login and run `curl -O https://inspector-agent.amazonaws.com/linux/latest/install` then `sudo bash install` to install the latest inspector agent.

## CodeCommit Permissions

When you create a branch in an AWS CodeCommit repository, the branch is available, by default, **to all repository users**.

We’ll show you how to create a policy in IAM that **prevents** users from pushing commits to and merging pull requests **to a branch named master**. 

Create users in IAM， and add users to defined groups, and add policies.

**Create a policy** in IAM that will deny API actions if certain conditions are met. We want to prevent users with this policy applied from updating a branch named master.

**Create a policy** in IAM that will deny API actions if certain conditions are met. We want to prevent users with this policy applied from updating a branch named master, but we don’t want to prevent them from viewing the branch, cloning the repository, or creating pull requests that will merge to that branch.

## Complete CI/CD with AWS

* AWS CodeCommit – A fully-managed source control service that hosts secure Git-based repositories, similar to GitHub or GitLab.

* AWS CodeBuild – A fully managed continuous integration service that compiles source code, runs tests, and produces software packages that are ready to deploy, on a dynamically created build server. 

* AWS CodeDeploy – A fully managed deployment service that automates software deployments to a variety of compute services such as Amazon EC2, AWS Fargate, AWS Lambda, and your on-premises servers. This solution uses CodeDeploy to deploy the code or application onto a set of EC2 instances running CodeDeploy agents.

* AWS CodePipeline –  This solution creates an end-to-end pipeline that fetches the application code from CodeCommit, builds and tests using CodeBuild, and finally deploys using CodeDeploy.

* AWS CloudWatch Events – An AWS CloudWatch Events rule is created to trigger the CodePipeline on a Git commit to the CodeCommit repository.

* Amazon Simple Storage Service (Amazon S3) – This solution stores the build and deployment artifacts created during the pipeline run.

* AWS Key Management Service (AWS KMS) – AWS KMS makes it easy for you to create and manage cryptographic keys and control their use.

## AWS OpsWorks vs AWS Beanstalk

## Api Gateway Auth

Best practices of invoking an API include using API Gateway Authoerizer, which is a lambda that checks coming requests before functional lambda to be invoked. It creates a policy document stipulating what action is allowed. 

```js
export.handler = function(event, context,callback){
    callback(generateAllow(event))
}

var generateAllow =function(event) {
    var statementOne = {}
    statementOne.Action = 'execute-api:Invoke'
    statementOne.Effect = 'Allow'
    statementOne.Resource = event.methodArn

    var policyDoc = {}
    policyDoc.Version = '2012-10-17'
    policyDoc.Statement = []
    policyDoc.Statement[0] = statementOne

    var authResponse = {};
    authResponse.principlalId = null;
    authResponse.policyDocument = policyDoc

    return authResponse
}
```