import consul
import time, json
import requests
import atexit
from flask import Flask
import threading

app = Flask(__name__)

this_addr='127.0.0.1'
this_port=8001
this_service_name="service-a"
this_service_id=f'{this_service_name}-{this_addr}-{this_port}'

another_service_name="service-b"

@app.route('/health')
def health():
    return "OK", 200

@app.route('/hello')
def hello():
    return f"Hello from {this_service_name}", 200

# Define a function to deregister the service from Consul
def deregister_service():
    c.agent.service.deregister(this_service_id)
    print(f"Service {this_service_id} deregistered from Consul")

# Register the deregistration function to run when the app shuts down
atexit.register(deregister_service)

if __name__ == '__main__':
    # Start Flask server in a background thread for health checks
    threading.Thread(target=lambda: app.run(port=this_port), daemon=True).start()

    # Connect to Consul
    # Default: Consul(host='127.0.0.1', port=8500, token=None, scheme='http', consistency='default', dc=None, verify=True, cert=None)
    c = consul.Consul()

    # Register Task A with Consul
    c.agent.service.register(
        name=this_service_name,
        service_id=this_service_id,
        address=this_addr,
        port=this_port,
        check=consul.Check.http(f'http://{this_addr}:{this_port}/health', interval='10s')
    )

    c.kv.put('config/feature_x_config', '{"is_enabled": true}')
    time.sleep(1)
    feature_x_config = json.loads(c.kv.get('config/feature_x_config')[1]['Value'].decode('utf-8'))
    if feature_x_config.get("is_enabled") == True:
        print("This feature feature_x_enabled is enabled.")

    # Discover Task B and send HTTP request
    while True:
        services = c.health.service(another_service_name, passing=True)[1]
        if services:
            found_another_service_address = services[0]['Service']['Address']
            found_another_service_port = services[0]['Service']['Port']
            url = f"http://{found_another_service_address}:{found_another_service_port}/hello"
            response = requests.get(url)
            print(f"Response from {another_service_name}: {response.content.decode()}")
        else:
            print(f"No healthy {another_service_name} found")
        time.sleep(5)