from pyramid.view import view_config

project_name = "pyramid_tutorial"

# First view, available at http://localhost:6543/
@view_config(route_name='home', renderer=project_name+':' + 'templates/home.pt')
def home(request):
    client = request.headers["Host"]
    return {'name': client}

