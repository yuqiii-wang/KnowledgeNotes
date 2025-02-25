// Import utility functions from utils.groovy
def utils = load 'utils.groovy'

// Define a function to execute the pipeline
def executePipeline(Map config) {
    node('any') {
        checkout scm
        timestamps {
            stage('Build') {
                echo '🔨 Building the application...'
                utils.printConfig(config) // Use utility function
                mvnCommand('clean compile package')
            }
            stage('Test') {
                echo '🧪 Running tests...'
                mvnCommand('test')
            }
            stage('Archive') {
                echo '📦 Archiving artifacts...'
                archiveArtifacts artifacts: 'target/*.jar', fingerprint: true
            }
        }
    }
}

// Helper function to run Maven commands
def mvnCommand(String args) {
    sh "mvn ${args}"
}

// Return the executePipeline function to be called from Jenkinsfile
return this