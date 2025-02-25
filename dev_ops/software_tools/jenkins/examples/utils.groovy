// Utility function to print configuration
def printConfig(Map config) {
    echo "📄 Configuration:"
    config.each { key, value ->
        echo "${key}: ${value}"
    }
}

// Return utility functions
return this