# Ansible

Ansible is a deployment tool that runs a declared playbook (contains bash scripts).

## Core Components of Ansible

* Inventory: A list of server hosts that Ansible manages.
* Roles: A way to organize playbooks and other files to facilitate reuse and sharing.* Playbooks: YAML files that describe the desired state of the system and the tasks to achieve that state.
* Tasks: Individual units of work/step in a playbook. Each task calls an Ansible module.
* Modules: The units of work/tool that Ansible ships out to remote machines, e.g., `git` to pull built code, `command` to run bash

For example,

```yaml
- name: Clone the Flask application repository
  git:
    repo: https://github.com/yourusername/your-flask-app-repo.git
    dest: /opt/flask-app
    version: main  # Replace with your branch name if different

- name: Create a virtual environment
  command: python3 -m venv /opt/flask-app/venv
  args:
    creates: /opt/flask-app/venv
```

## Example

to deploy a python FLASK app, the role will be named `flask_app`, and its directory structure will look like this:

```txt
roles/
  flask_app/
    tasks/
      main.yml           # Contains all tasks for deploying the Flask app
    handlers/
      main.yml           # Handlers (if needed, e.g., restarting services)
    files/
      flask-app.service  # Systemd service file for the Flask app
    templates/           # Jinja2 templates (if needed)
    vars/
      main.yml           # Variables specific to the role
    defaults/
      main.yml           # Default variables (lowest precedence)
    meta/
      main.yml           # Metadata (e.g., role dependencies)
```

The playbook has

```yaml
# roles/flask_app/tasks/main.yml
---
- name: Update apt cache
  apt:
    update_cache: yes

- name: Install required packages
  apt:
    name:
      - python3
      - python3-pip
      - python3-venv
      - git
    state: present

- name: Clone the Flask application repository
  git:
    repo: https://github.com/yourusername/your-flask-app-repo.git
    dest: /opt/flask-app
    version: main  # Replace with your branch name if different

- name: Create a virtual environment
  command: python3 -m venv /opt/flask-app/venv
  args:
    creates: /opt/flask-app/venv

- name: Install Flask application dependencies
  pip:
    requirements: /opt/flask-app/requirements.txt
    virtualenv: /opt/flask-app/venv

- name: Copy systemd service file for the Flask application
  copy:
    src: flask-app.service
    dest: /etc/systemd/system/flask-app.service
    owner: root
    group: root
    mode: '0644'

- name: Reload systemd daemon
  systemd:
    daemon_reload: yes

- name: Enable and start the Flask application service
  systemd:
    name: flask-app
    state: started
    enabled: yes

- name: Allow traffic on port 5000 (Flask default port)
  ufw:
    rule: allow
    port: 5000
    proto: tcp
```
